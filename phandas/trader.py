"""OKX cryptocurrency perpetual swap trading and portfolio rebalancing."""

import time
import string
import random
import pandas as pd
from typing import Dict, List, Optional

from rich.box import SIMPLE
from .console import print, console, Table


def _generate_client_order_id() -> str:
    timestamp = str(int(time.time() * 1000))
    random_suffix = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
    return f"t{timestamp}{random_suffix}"[:32]


def _safe_float(val: object) -> float:
    try:
        return float(val) if val and val != '' else 0.0
    except (ValueError, TypeError):
        return 0.0


class OKXTrader:
    """OKX perpetual swap trading interface."""
    
    def __init__(self, api_key: str, secret_key: str, passphrase: str, 
                 use_testnet: bool = True, inst_type: str = 'SWAP'):
        import okx.Account as Account
        import okx.Trade as Trade
        import okx.MarketData as MarketData
        
        self.api_key = api_key
        self.secret_key = secret_key
        self.passphrase = passphrase
        self.use_testnet = use_testnet
        self.inst_type = inst_type
        self.flag = '1' if use_testnet else '0'
        
        self.account_api = Account.AccountAPI(api_key, secret_key, passphrase, False, self.flag)
        self.trade_api = Trade.TradeAPI(api_key, secret_key, passphrase, False, self.flag)
        self.market_api = MarketData.MarketAPI(api_key, secret_key, passphrase, False, self.flag)
        
        self.acct_lv = None
        self.pos_mode = None
    
    def validate_account_config(self) -> Dict:
        """Validate account configuration before trading."""
        config = self.get_account_config()
        
        if 'error' in config:
            return {'status': 'error', 'msg': f"Cannot get account config: {config['error']}"}
        
        acct_lv_str = config.get('acct_lv')
        pos_mode = config.get('pos_mode')
        acct_lv_map_reverse = {'SPOT': '1', 'FUTURES': '2', 'CROSS_MARGIN': '3', 'COMPOSITE': '4'}
        acct_lv_code = acct_lv_map_reverse.get(acct_lv_str, '')
        
        checks = {
            'account_mode': {
                'value': f"{acct_lv_str} ({acct_lv_code})",
                'required': ['2', '3'],
                'status': 'ok' if acct_lv_code in ['2', '3'] else 'error'
            },
            'position_mode': {
                'value': 'net_mode (one-way)' if pos_mode == 'net_mode' else 'long_short_mode (hedge)',
                'required': 'net_mode',
                'status': 'ok' if pos_mode == 'net_mode' else 'error'
            }
        }
        
        errors = []
        if checks['account_mode']['status'] == 'error':
            errors.append(f"Account mode must be FUTURES(2) or CROSS_MARGIN(3), current: {checks['account_mode']['value']}")
        if checks['position_mode']['status'] == 'error':
            errors.append(f"Position mode must be net_mode (one-way), current: {checks['position_mode']['value']}")
        
        self.acct_lv = acct_lv_code
        self.pos_mode = pos_mode
        
        if errors:
            return {'status': 'error', 'msg': '; '.join(errors), 'checks': checks}
        
        return {'status': 'ok', 'msg': 'Account config validated', 'checks': checks, 
                'acct_lv': acct_lv_code, 'pos_mode': pos_mode}
    
    def get_positions(self, inst_id: Optional[str] = None) -> Dict[str, Dict]:
        """Get current positions filtered by instrument."""
        res = self.account_api.get_positions(self.inst_type, inst_id)
        if res['code'] != '0' or not res['data']:
            return {}
        
        positions = {}
        for pos in res['data']:
            pos_qty = float(pos.get('pos', 0))
            notional_usd = float(pos.get('notionalUsd', 0))
            mark_px = float(pos.get('markPx', 0))
            
            if pos_qty == 0 or notional_usd == 0 or mark_px == 0 or abs(notional_usd) < 0.01:
                continue
            
            inst = pos.get('instId')
            pos_side = pos.get('posSide')
            notional_usd = abs(notional_usd) if pos_qty > 0 else -abs(notional_usd)
            
            positions[f"{inst}_{pos_side}"] = {
                'symbol': inst,
                'pos_side': pos_side,
                'pos_qty': pos_qty,
                'mark_px': mark_px,
                'entry_px': float(pos.get('avgPx', 0)),
                'unrealized_pnl': float(pos.get('upl', 0)),
                'realized_pnl': float(pos.get('realizedPnl', 0)),
                'leverage': float(pos.get('lever', 1)),
                'maint_margin_ratio': float(pos.get('maintMarginRatio', 0)),
                'notional_usd': notional_usd
            }
        
        return positions
    
    def get_ticker(self, inst_id: str) -> Dict:
        """Get market ticker data."""
        res = self.market_api.get_ticker(inst_id)
        if res['code'] != '0' or not res['data']:
            return {'error': res.get('msg', 'Unknown error')}
        
        ticker = res['data'][0]
        return {
            'symbol': ticker.get('instId'),
            'last_px': float(ticker.get('last', 0)),
            'bid_px': float(ticker.get('bidPx', 0)),
            'ask_px': float(ticker.get('askPx', 0)),
            'vol_24h': float(ticker.get('vol24h', 0)),
            'high_24h': float(ticker.get('high24h', 0)),
            'low_24h': float(ticker.get('low24h', 0)),
            'timestamp': ticker.get('ts')
        }
    
    def get_instrument_info(self, inst_id: str) -> Dict:
        """Get instrument specifications."""
        res = self.account_api.get_instruments(instType=self.inst_type, instId=inst_id)
        if res['code'] != '0' or not res['data']:
            return {'error': res.get('msg', 'Unknown error')}
        
        inst = res['data'][0]
        
        return {
            'symbol': inst.get('instId'),
            'state': inst.get('state'),
            'min_sz': _safe_float(inst.get('minSz')),
            'tick_sz': _safe_float(inst.get('tickSz')),
            'lot_sz': _safe_float(inst.get('lotSz')),
            'max_lmt_sz': _safe_float(inst.get('maxLmtSz')),
            'max_mkt_sz': _safe_float(inst.get('maxMktSz')),
            'max_mkt_amt': _safe_float(inst.get('maxMktAmt')),
            'ct_mult': _safe_float(inst.get('ctMult')),
            'ct_val': _safe_float(inst.get('ctVal')),
        }
    
    def set_leverage(self, inst_id: str, lever: int, mgn_mode: str = 'cross', 
                     pos_side: str = 'long') -> Dict:
        """Set leverage for instrument."""
        if lever < 1 or lever > 125:
            return {'status': 'error', 'msg': f"Invalid leverage: {lever}"}
        
        leverage_params = {
            'instId': inst_id,
            'lever': str(lever),
            'mgnMode': mgn_mode
        }
        
        if self.pos_mode and self.pos_mode != 'net_mode' and mgn_mode == 'isolated':
            leverage_params['posSide'] = pos_side
        
        res = self.account_api.set_leverage(**leverage_params)
        
        if res['code'] == '0' and res['data']:
            data = res['data'][0]
            return {
                'inst_id': data.get('instId'),
                'lever': data.get('lever'),
                'mgn_mode': data.get('mgnMode'),
                'pos_side': data.get('posSide', 'N/A'),
                'status': 'success',
                'msg': 'Leverage set successfully'
            }
        return {'status': 'error', 'msg': res.get('msg', 'Unknown error'), 'code': res.get('code')}
    
    def convert_coin_contract(self, inst_id: str, sz: float, convert_type: int = 1, 
                            px: Optional[float] = None, unit: str = 'coin') -> Dict:
        """Convert between coin and contract sizes."""
        import okx.PublicData as PublicData
        
        if unit == 'usds' and px is None:
            ticker = self.get_ticker(inst_id)
            if 'error' in ticker:
                return {'status': 'error', 'msg': f"Cannot get price for {inst_id}: {ticker.get('error')}"}
            px = ticker.get('last_px')
        
        public_api = PublicData.PublicAPI(flag=self.flag)
        
        params = {
            'instId': inst_id,
            'sz': str(sz),
            'type': str(convert_type),
            'unit': unit
        }
        if px is not None:
            params['px'] = str(px)
        
        res = public_api.get_convert_contract_coin(**params)
        if res['code'] != '0' or not res['data']:
            return {'status': 'error', 'msg': res.get('msg', 'Unknown error')}
        
        data = res['data'][0]
        return {
            'inst_id': data.get('instId'),
            'sz': float(data.get('sz', 0)),
            'px': float(data.get('px', 0)) if data.get('px') else None,
            'convert_type': int(data.get('type', convert_type)),
            'unit': data.get('unit', unit),
            'status': 'success',
            'msg': f"Convert {sz} to {data.get('sz')} (type={convert_type})"
        }
    
    def place_order(self, inst_id: str, side: str, size: float, 
                    price: Optional[float] = None, 
                    pos_side: str = 'long',
                    reduce_only: bool = False,
                    _pos_mode: Optional[str] = None) -> Dict:
        """Place a single order."""
        ord_type = 'limit' if price else 'market'
        client_order_id = _generate_client_order_id()
        td_mode = 'cash' if self.inst_type == 'SPOT' else 'cross'
        
        order_params = {
            'instId': inst_id,
            'tdMode': td_mode,
            'side': side,
            'ordType': ord_type,
            'sz': str(size),
            'clOrdId': client_order_id,
            'tag': '82eebde453a2BCDE',
        }
        
        if self.inst_type in ['SWAP', 'FUTURES']:
            if _pos_mode is None:
                account_config = self.get_account_config()
                _pos_mode = account_config.get('pos_mode')
            
            order_params['posSide'] = 'net' if _pos_mode == 'net_mode' else pos_side
            order_params['reduceOnly'] = 'true' if reduce_only else 'false'
        elif not price:
            order_params['tgtCcy'] = 'quote_ccy'
        
        if price:
            order_params['px'] = str(price)
        
        res = self.trade_api.place_order(**order_params)
        
        if res['code'] == '0' and res['data']:
            return {
                'order_id': res['data'][0].get('ordId'),
                'client_order_id': client_order_id,
                'status': 'success',
                'msg': 'Order placed successfully'
            }
        return {'status': 'error', 'msg': res.get('msg', 'Unknown error'), 'code': res.get('code')}
    
    def place_batch_orders(self, orders: List[Dict]) -> Dict:
        """Place multiple orders (max 20 per request)."""
        if not orders:
            return {'status': 'error', 'msg': 'Orders list is empty'}
        if len(orders) > 20:
            return {'status': 'error', 'msg': f'Too many orders: {len(orders)}, max is 20'}
        
        batch_orders = []
        account_config = self.get_account_config()
        _pos_mode = account_config.get('pos_mode')
        
        for order in orders:
            inst_id = order.get('inst_id')
            side = order.get('side')
            size = order.get('size')
            price = order.get('price')
            pos_side = order.get('pos_side', 'long')
            reduce_only = order.get('reduce_only', False)
            
            ord_type = 'limit' if price else 'market'
            client_order_id = _generate_client_order_id()
            td_mode = 'cash' if self.inst_type == 'SPOT' else 'cross'
            
            order_params = {
                'instId': inst_id,
                'tdMode': td_mode,
                'side': side,
                'ordType': ord_type,
                'sz': str(size),
                'clOrdId': client_order_id,
                'tag': '82eebde453a2BCDE',
            }
            
            if self.inst_type in ['SWAP', 'FUTURES']:
                order_params['posSide'] = 'net' if _pos_mode == 'net_mode' else pos_side
                order_params['reduceOnly'] = 'true' if reduce_only else 'false'
            elif not price:
                order_params['tgtCcy'] = 'quote_ccy'
            
            if price:
                order_params['px'] = str(price)
            
            batch_orders.append(order_params)
        
        res = self.trade_api.place_multiple_orders(batch_orders)
        
        if res['code'] == '0' and res['data']:
            results = [
                {
                    'order_id': item.get('ordId'),
                    'client_order_id': item.get('clOrdId'),
                    'status': 'success' if item.get('sCode') == '0' else 'error',
                    'msg': item.get('sMsg', '')
                }
                for item in res['data']
            ]
            
            successful = sum(1 for r in results if r['status'] == 'success')
            failed = len(results) - successful
            
            return {
                'status': 'success' if failed == 0 else ('partial' if successful > 0 else 'error'),
                'total': len(results),
                'successful': successful,
                'failed': failed,
                'orders': results
            }
        
        return {'status': 'error', 'msg': res.get('msg', 'Unknown error'), 'code': res.get('code')}
    
    def get_account_balance_info(self) -> Dict:
        """Get account balance information."""
        res = self.account_api.get_account_balance()
        if res['code'] != '0' or not res['data']:
            return {'error': res.get('msg', 'Unknown error')}
        
        data = res['data'][0]
        details = data.get('details', [])
        usdt_info = next((d for d in details if d.get('ccy') == 'USDT'), {})
        
        return {
            'total_equity': _safe_float(data.get('totalEq', 0)),
            'available_equity': _safe_float(data.get('availEq', 0)),
            'used_margin': _safe_float(data.get('imr', 0)),
            'maint_margin': _safe_float(data.get('mmr', 0)),
            'unrealized_pnl': _safe_float(data.get('upl', 0)),
            'usdt_balance': _safe_float(usdt_info.get('cashBal', 0)),
            'usdt_available': _safe_float(usdt_info.get('availBal', 0)),
            'usdt_frozen': _safe_float(usdt_info.get('frozenBal', 0)),
            'details': details
        }
    
    def get_account_config(self) -> Dict:
        """Get account configuration."""
        res = self.account_api.get_account_config()
        if res['code'] != '0' or not res['data']:
            return {'error': res.get('msg', 'Unknown error')}
        
        config = res['data'][0]
        acct_lv = config.get('acctLv')
        acct_lv_map = {'1': 'SPOT', '2': 'FUTURES', '3': 'CROSS_MARGIN', '4': 'COMPOSITE'}
        
        return {
            'acct_lv': acct_lv_map.get(acct_lv, acct_lv),
            'pos_mode': config.get('posMode'),
            'auto_loan': config.get('autoLoan', False),
            'kyc_lv': config.get('kycLv'),
            'uid': config.get('uid'),
            'level': config.get('level'),
            'opAuth': config.get('opAuth'),
            'settlement_ccy': config.get('settleCcy'),
            'spot_offset_type': config.get('spotOffsetType'),
            'greek_type': config.get('greeksType')
        }

    def close_position(self, inst_id: str, mgn_mode: str = 'cross', 
                      pos_side: Optional[str] = None, 
                      auto_cxl: bool = False) -> Dict:
        """Close position at market price."""
        close_params = {'instId': inst_id, 'mgnMode': mgn_mode}
        if pos_side:
            close_params['posSide'] = pos_side
        if auto_cxl:
            close_params['autoCxl'] = 'true'
        
        res = self.trade_api.close_positions(**close_params)
        
        if res['code'] == '0' and res['data']:
            data = res['data'][0]
            return {
                'inst_id': data.get('instId'),
                'pos_side': data.get('posSide'),
                'status': 'success',
                'msg': 'Position closed successfully'
            }
        return {'status': 'error', 'msg': res.get('msg', 'Unknown error'), 'code': res.get('code')}
    
    def close_all_positions(self, mgn_mode: str = 'cross', 
                           symbol_suffix: str = '-USDT-SWAP') -> Dict:
        """Close all positions matching symbol suffix."""
        positions = self.get_positions()
        
        if not positions:
            return {'status': 'success', 'msg': 'No positions to close', 
                   'total': 0, 'closed': 0, 'failed': 0, 'results': []}
        
        account_config = self.get_account_config()
        pos_mode = account_config.get('pos_mode', 'net_mode')
        
        close_results = []
        successful = 0
        failed = 0
        
        for pos_key, pos_data in positions.items():
            inst_id = pos_data['symbol']
            pos_side = pos_data['pos_side']
            
            if symbol_suffix and not inst_id.endswith(symbol_suffix):
                continue
            
            try:
                close_side = None if pos_mode == 'net_mode' else pos_side
                result = self.close_position(inst_id=inst_id, mgn_mode=mgn_mode, 
                                            pos_side=close_side, auto_cxl=False)
                
                if result['status'] == 'success':
                    successful += 1
                else:
                    failed += 1
                
                close_results.append({
                    'inst_id': inst_id,
                    'pos_side': pos_side,
                    'status': result['status'],
                    'msg': result.get('msg', '')
                })
                
                time.sleep(0.1)
            
            except Exception as e:
                failed += 1
                close_results.append({
                    'inst_id': inst_id,
                    'pos_side': pos_side,
                    'status': 'error',
                    'msg': str(e)
                })
        
        return {
            'status': 'success' if failed == 0 else ('partial' if successful > 0 else 'error'),
            'total': len(close_results),
            'closed': successful,
            'failed': failed,
            'results': close_results,
            'msg': f'Closed {successful} positions, {failed} failed'
        }

    def rebalance_portfolio(self, target_weights: Dict[str, float], 
                           budget: Optional[float] = None,
                           symbol_suffix: str = '-USDT-SWAP',
                           leverage: int = 5) -> Dict:
        """Rebalance portfolio to target weights."""
        validation = self.validate_account_config()
        if validation['status'] == 'error':
            return {'status': 'error', 'msg': validation['msg'], 'validation': validation['checks']}
        
        if not target_weights or budget is None or budget <= 0:
            return {'status': 'error', 
                   'msg': 'Target weights and budget required. Call get_account_balance_info() for total_equity.'}
        
        base_budget = budget
        current_positions = self.get_positions()
        current_holdings = {}
        
        for pos_key, pos_data in current_positions.items():
            symbol = pos_data['symbol'].split('-')[0]
            notional_usd = pos_data['notional_usd']
            current_holdings[symbol] = {
                'inst_id': pos_data['symbol'],
                'side': 'long' if notional_usd > 0 else 'short',
                'qty': pos_data['pos_qty'],
                'mark_px': pos_data['mark_px'],
                'entry_px': pos_data['entry_px'],
                'usd_value': notional_usd
            }
        
        target_holdings = {
            symbol: {
                'weight': weight,
                'target_usd': weight * base_budget,
                'direction': 'long' if weight > 0 else ('short' if weight < 0 else 'none'),
            }
            for symbol, weight in target_weights.items()
        }
        
        rebalance_trades = []
        orders_to_execute = []
        all_symbols = set(target_holdings.keys()) | set(current_holdings.keys())
        
        for symbol in all_symbols:
            inst_id = f"{symbol}{symbol_suffix}"
            
            current = current_holdings.get(symbol, {
                'inst_id': inst_id,
                'side': None,
                'qty': 0,
                'mark_px': 0,
                'usd_value': 0
            })
            
            target = target_holdings.get(symbol, {
                'weight': 0,
                'target_usd': 0,
                'direction': 'none'
            })
            
            current_usd = current['usd_value']
            target_usd = target['target_usd']
            diff_usd = target_usd - current_usd
            
            if abs(diff_usd) < 0.01:
                action = 'skip'
                order_side = None
                order_size = 0
            elif diff_usd > 0:
                action = 'long'
                order_side = 'buy'
                order_size = diff_usd
            else:
                action = 'short'
                order_side = 'sell'
                order_size = abs(diff_usd)
            
            trade_record = {
                'symbol': symbol,
                'target_weight': target['weight'],
                'target_usd': target_usd,
                'current_usd': current_usd,
                'diff_usd': diff_usd,
                'action': action,
                'order_size': order_size,
                'order_id': None,
                'status': 'pending',
                'msg': None
            }
            
            if action != 'skip' and order_side and order_size > 0:
                orders_to_execute.append({
                    'symbol': symbol,
                    'inst_id': inst_id,
                    'side': order_side,
                    'usd_amount': order_size,
                    'trade_record': trade_record
                })
            
            rebalance_trades.append(trade_record)
        
        successful_orders = 0
        failed_orders = 0
        prepared_orders = []
        order_to_trade_mapping = {}
        account_config = self.get_account_config()
        _pos_mode = account_config.get('pos_mode')
        
        for order_info in orders_to_execute:
            inst_id = order_info['inst_id']
            side = order_info['side']
            usd_amount = order_info['usd_amount']
            trade_record = order_info['trade_record']
            symbol = order_info['symbol']
            
            try:
                lever_result = self.set_leverage(inst_id=inst_id, lever=leverage, 
                                                mgn_mode='cross', 
                                                pos_side='long' if side == 'buy' else 'short')
                
                if lever_result['status'] != 'success':
                    trade_record['status'] = 'error'
                    trade_record['msg'] = f"Failed to set leverage: {lever_result.get('msg', 'Unknown error')}"
                    failed_orders += 1
                    continue
                
                ticker = self.get_ticker(inst_id)
                current_px = ticker.get('last_px') if 'error' not in ticker else None
                
                convert_result = self.convert_coin_contract(inst_id=inst_id, sz=usd_amount,
                                                           convert_type=1, px=current_px, unit='usds')
                
                time.sleep(0.25)
                
                if convert_result['status'] != 'success':
                    trade_record['status'] = 'error'
                    trade_record['msg'] = convert_result['msg']
                    failed_orders += 1
                    continue
                
                contract_size = convert_result['sz']
                
                if contract_size <= 0:
                    trade_record['status'] = 'error'
                    trade_record['msg'] = f"USD amount {usd_amount} is too small"
                    failed_orders += 1
                    continue
                
                order_params = {
                    'inst_id': inst_id,
                    'side': side,
                    'size': contract_size,
                    'price': None,
                    'pos_side': 'long' if side == 'buy' else 'short',
                    'reduce_only': trade_record.get('action') in ['reduce', 'close']
                }
                
                prepared_orders.append(order_params)
                order_to_trade_mapping[len(prepared_orders) - 1] = {
                    'trade_record': trade_record,
                    'symbol': symbol
                }
            
            except Exception as e:
                trade_record['status'] = 'error'
                trade_record['msg'] = str(e)
                failed_orders += 1
        
        if prepared_orders:
            batch_result = self.place_batch_orders(prepared_orders)
            
            if batch_result['status'] != 'error' and 'orders' in batch_result:
                for i, order_result in enumerate(batch_result['orders']):
                    mapping = order_to_trade_mapping.get(i)
                    if not mapping:
                        continue
                    
                    trade_record = mapping['trade_record']
                    
                    if order_result['status'] == 'success':
                        trade_record['order_id'] = order_result.get('order_id')
                        trade_record['status'] = 'success'
                        successful_orders += 1
                    else:
                        trade_record['status'] = 'error'
                        trade_record['msg'] = order_result.get('msg', 'Unknown error')
                        failed_orders += 1
            else:
                for i in order_to_trade_mapping:
                    trade_record = order_to_trade_mapping[i]['trade_record']
                    trade_record['status'] = 'error'
                    trade_record['msg'] = batch_result.get('msg', 'Batch order failed')
                    failed_orders += 1
        
        symbols_rebalanced = sum(1 for t in rebalance_trades if t['status'] == 'success')
        symbols_error = sum(1 for t in rebalance_trades if t['status'] == 'error')
        
        return {
            'status': 'success' if failed_orders == 0 else ('partial' if successful_orders > 0 else 'error'),
            'budget': base_budget,
            'total_position_usd': sum(abs(h['usd_value']) for h in current_holdings.values()),
            'rebalance_trades': rebalance_trades,
            'summary': {
                'symbols_rebalanced': symbols_rebalanced,
                'symbols_error': symbols_error,
                'total_orders': len(prepared_orders),
                'successful_orders': successful_orders,
                'failed_orders': failed_orders
            },
            'msg': f'Rebalance completed: {successful_orders} success, {failed_orders} failed'
        }


class Rebalancer:
    """High-level portfolio rebalancing interface."""
    
    def __init__(self, target_weights: Dict[str, float], trader: OKXTrader,
                 budget: Optional[float] = None, symbol_suffix: str = '-USDT-SWAP',
                 leverage: int = 5, preview: bool = False):
        self.target_weights = target_weights
        self.trader = trader
        self.budget = budget
        self.symbol_suffix = symbol_suffix
        self.leverage = leverage
        self.preview_mode = preview
        self.result = None
        
        self.plan_data = None
        self.current_holdings = None
        self.target_holdings = None
        self.all_symbols = None
    
    def plan(self) -> 'Rebalancer':
        """Generate rebalancing plan without executing orders."""
        if not self.target_weights:
            raise ValueError('Target weights is empty')
        if self.budget is None or self.budget <= 0:
            raise ValueError('Budget must be specified and greater than 0')
        
        current_positions = self.trader.get_positions()
        self.current_holdings = {}
        
        for pos_key, pos_data in current_positions.items():
            symbol = pos_data['symbol'].split('-')[0]
            notional_usd = pos_data['notional_usd']
            self.current_holdings[symbol] = {
                'inst_id': pos_data['symbol'],
                'side': 'long' if notional_usd > 0 else 'short',
                'qty': pos_data['pos_qty'],
                'mark_px': pos_data['mark_px'],
                'entry_px': pos_data['entry_px'],
                'usd_value': notional_usd
            }
        
        self.target_holdings = {
            symbol: {
                'weight': weight,
                'target_usd': weight * self.budget,
                'direction': 'long' if weight > 0 else ('short' if weight < 0 else 'none'),
            }
            for symbol, weight in self.target_weights.items()
        }
        
        self.all_symbols = set(self.target_holdings.keys()) | set(self.current_holdings.keys())
        
        plan_trades = []
        for symbol in sorted(self.all_symbols):
            current = self.current_holdings.get(symbol, {
                'inst_id': f"{symbol}{self.symbol_suffix}",
                'side': None,
                'qty': 0,
                'mark_px': 0,
                'usd_value': 0
            })
            
            target = self.target_holdings.get(symbol, {
                'weight': 0,
                'target_usd': 0,
                'direction': 'none'
            })
            
            current_usd = current['usd_value']
            target_usd = target['target_usd']
            diff_usd = target_usd - current_usd
            
            if abs(diff_usd) < 1.0:
                action = "skip"
            elif current_usd == 0 and target_usd == 0:
                action = "none"
            elif current_usd == 0 and target_usd != 0:
                action = "open"
            elif current_usd > 0 and target_usd > 0:
                action = "add" if diff_usd > 0 else "reduce"
            elif current_usd < 0 and target_usd < 0:
                action = "add" if diff_usd < 0 else "reduce"
            elif (current_usd > 0 and target_usd < 0) or (current_usd < 0 and target_usd > 0):
                action = "flip"
            else:
                action = "close"
            
            plan_trades.append({
                'symbol': symbol,
                'current_usd': current_usd,
                'target_usd': target_usd,
                'diff_usd': diff_usd,
                'action': action,
                'weight': target['weight']
            })
        
        self.plan_data = plan_trades
        return self
    
    @property
    def plan_df(self) -> pd.DataFrame:
        if not self.plan_data:
            return pd.DataFrame()
        return pd.DataFrame(self.plan_data)
    
    def print_preview(self) -> 'Rebalancer':
        """Print rebalancing preview table."""
        if not self.plan_data:
            raise ValueError('Plan not generated. Call plan() first.')
        
        current_position_total = sum(abs(h['usd_value']) for h in self.current_holdings.values())
        
        console.print(f"\n[bold]Rebalancer[/bold]  equity=[bright_green]${self.budget:,.2f}[/bright_green]  position=[bright_yellow]${current_position_total:,.2f}[/bright_yellow]\n")
        
        weights_table = Table(title="[bold]Target Weights[/bold]", box=SIMPLE, show_header=True, header_style="bold")
        weights_table.add_column("Symbol", style="bright_cyan")
        weights_table.add_column("Weight", justify="right")
        for symbol, weight in sorted(self.target_weights.items()):
            style = "bright_green" if weight > 0 else "bright_red"
            weights_table.add_row(symbol, f"[{style}]{weight:+.4f}[/{style}]")
        console.print(weights_table)
        console.print()
        
        holdings_table = Table(title="[bold]Holdings[/bold]", box=SIMPLE, show_header=True, header_style="bold")
        holdings_table.add_column("Symbol", style="bright_cyan")
        holdings_table.add_column("Current", justify="right")
        holdings_table.add_column("Target", justify="right")
        holdings_table.add_column("Delta", justify="right")
        holdings_table.add_column("Action", justify="center")
        
        def fmt_usd(val):
            if val >= 0:
                return f"${val:,.2f}"
            else:
                return f"-${abs(val):,.2f}"
        
        for trade in self.plan_data:
            delta = trade['diff_usd']
            if delta > 0.01:
                delta_style = "bright_cyan"
            elif delta < -0.01:
                delta_style = "bright_magenta"
            else:
                delta_style = "dim"
            delta_str = f"+${delta:,.2f}" if delta >= 0 else f"-${abs(delta):,.2f}"
            
            action = trade['action']
            if action == 'skip':
                action_str = "[dim]skip[/dim]"
            elif action == 'long':
                action_str = "[bold bright_green]LONG[/bold bright_green]"
            else:
                action_str = "[bold bright_red]SHORT[/bold bright_red]"
            
            holdings_table.add_row(
                trade['symbol'],
                fmt_usd(trade['current_usd']),
                fmt_usd(trade['target_usd']),
                f"[{delta_style}]{delta_str}[/{delta_style}]",
                action_str,
            )
        
        current_abs_sum = sum(abs(trade['current_usd']) for trade in self.plan_data)
        target_abs_sum = sum(abs(trade['target_usd']) for trade in self.plan_data)
        diff_abs_sum = sum(abs(trade['diff_usd']) for trade in self.plan_data)
        holdings_table.add_section()
        holdings_table.add_row(
            "[bold]Total[/bold]",
            f"[bold]{fmt_usd(current_abs_sum)}[/bold]",
            f"[bold]{fmt_usd(target_abs_sum)}[/bold]",
            f"[bold]{fmt_usd(diff_abs_sum)}[/bold]",
            "",
        )
        
        console.print(holdings_table)
        
        return self
    
    def run(self) -> 'Rebalancer':
        """Execute rebalancing with optional preview and confirmation."""
        if self.preview_mode:
            self.plan()
            self.print_preview()
            
            try:
                input("\nPress Enter to execute, Ctrl+C to cancel: ")
            except KeyboardInterrupt:
                print("Rebalancing cancelled by user")
                return self
        
        self.result = self.trader.rebalance_portfolio(
            target_weights=self.target_weights,
            budget=self.budget,
            symbol_suffix=self.symbol_suffix,
            leverage=self.leverage
        )
        return self
    
    def summary(self) -> str:
        """Generate rebalancing summary in scikit-learn style."""
        if not self.result:
            return "Rebalancer(not executed, call run() first)"
        
        r = self.result
        s = r['summary']
        
        lines = [
            f"Rebalancer(status={r['status']}, budget=${r['budget']:,.2f})",
            f"  orders:  total={s['total_orders']}  success={s['successful_orders']}  failed={s['failed_orders']}",
            f"  trades:",
        ]
        
        for trade in r['rebalance_trades']:
            status_map = {'success': 'OK', 'partial': 'PARTIAL', 'skip': 'SKIP', 'error': 'ERROR', 'pending': 'PENDING'}
            status_str = status_map.get(trade['status'], trade['status'])
            if trade['action'] == 'none':
                status_str = 'SKIP'
            
            lines.append(
                f"    [{status_str:7}] {trade['symbol']:6}  "
                f"target=${trade['target_usd']:>10.2f}  "
                f"current=${trade['current_usd']:>10.2f}  "
                f"delta=${trade['diff_usd']:>+10.2f}  "
                f"action={trade['action']}"
            )
            
            if trade.get('msg'):
                lines.append(f"             msg={trade['msg']}")
        
        return "\n".join(lines)
    
    def print_summary(self) -> 'Rebalancer':
        """Print rebalancing summary."""
        print(self.summary())
        return self
    
    
    def __repr__(self) -> str:
        if not self.result:
            return (f"Rebalancer(budget={self.budget}, "
                   f"symbols={len(self.target_weights)}, status=not_executed)")
        return (f"Rebalancer(budget={self.budget}, "
               f"status={self.result['status']}, "
               f"success={self.result['summary']['successful_orders']}, "
               f"failed={self.result['summary']['failed_orders']})")


def rebalance(target_weights: Dict[str, float], trader: OKXTrader, budget: float,
             symbol_suffix: str = '-USDT-SWAP', leverage: int = 5, preview: bool = True, 
             auto_run: bool = True) -> Rebalancer:
    """
    Convenient rebalancing function.
    
    Parameters
    ----------
    target_weights : Dict[str, float]
        Target weight for each symbol
    trader : OKXTrader
        OKXTrader instance
    budget : float
        Total budget for rebalancing (required)
    symbol_suffix : str
        Symbol suffix (default: '-USDT-SWAP')
    leverage : int
        Leverage multiplier (default: 5)
    preview : bool
        Show rebalancing plan and wait for confirmation (default: True)
    auto_run : bool
        Automatically execute and return result (default: True).
        If False, returns Rebalancer instance for manual control.
    
    Returns
    -------
    Rebalancer
        Rebalancer instance with result (if auto_run=True, result is available immediately)
    
    Examples
    --------
    rb = rebalance(weights, trader, budget=10000)
    rb.print_summary()
    
    rb = rebalance(weights, trader, budget=10000, preview=False)
    rb.print_summary()
    
    rb = rebalance(weights, trader, budget=10000, auto_run=False)
    rb.plan().print_preview()
    rb.run()
    rb.print_summary()
    """
    rb = Rebalancer(target_weights, trader, budget, symbol_suffix, leverage, preview=preview)
    if auto_run:
        rb.run()
    return rb
