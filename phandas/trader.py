
import time
import string
import random
from typing import Dict, List, Optional

OKX_BROKER_CODE = '82eebde453a2BCDE'

def _generate_client_order_id() -> str:
    timestamp = str(int(time.time() * 1000))
    random_suffix = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
    return f"t{timestamp}{random_suffix}"[:32]


class OKXTrader:
    """OKX 永續合約交易接口"""
    
    def __init__(self, api_key: str, secret_key: str, passphrase: str, 
                 use_testnet: bool = True, inst_type: str = 'SWAP',
                 broker_code: Optional[str] = None):
        import okx.Account as Account
        import okx.Trade as Trade
        import okx.MarketData as MarketData
        
        self.api_key = api_key
        self.secret_key = secret_key
        self.passphrase = passphrase
        self.use_testnet = use_testnet
        self.inst_type = inst_type
        self.broker_code = broker_code or OKX_BROKER_CODE
        self.flag = '1' if use_testnet else '0'
        
        self.account_api = Account.AccountAPI(api_key, secret_key, passphrase, False, self.flag)
        self.trade_api = Trade.TradeAPI(api_key, secret_key, passphrase, False, self.flag)
        self.market_api = MarketData.MarketAPI(api_key, secret_key, passphrase, False, self.flag)
    
    def get_positions(self, inst_id: Optional[str] = None) -> Dict[str, Dict]:
        res = self.account_api.get_positions(self.inst_type, inst_id)
        if res['code'] != '0' or not res['data']:
            return {}
        
        positions = {}
        for pos in res['data']:
            inst = pos.get('instId')
            pos_side = pos.get('posSide')
            pos_qty = float(pos.get('pos', 0))
            notional_usd = float(pos.get('notionalUsd', 0))
            
            if pos_qty < 0:
                notional_usd = -abs(notional_usd)
            else:
                notional_usd = abs(notional_usd)
            
            positions[f"{inst}_{pos_side}"] = {
                'symbol': inst,
                'pos_side': pos_side,
                'pos_qty': pos_qty,
                'mark_px': float(pos.get('markPx', 0)),
                'entry_px': float(pos.get('avgPx', 0)),
                'unrealized_pnl': float(pos.get('upl', 0)),
                'realized_pnl': float(pos.get('realizedPnl', 0)),
                'leverage': float(pos.get('lever', 1)),
                'maint_margin_ratio': float(pos.get('maintMarginRatio', 0)),
                'notional_usd': notional_usd
            }
        
        return positions
    
    def get_ticker(self, inst_id: str) -> Dict:
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
        res = self.account_api.get_instruments(instType=self.inst_type, instId=inst_id)
        if res['code'] != '0' or not res['data']:
            return {'error': res.get('msg', 'Unknown error')}
        
        inst = res['data'][0]
        
        def safe_float(val):
            if not val or val == '':
                return None
            try:
                return float(val)
            except (ValueError, TypeError):
                return None
        
        return {
            'symbol': inst.get('instId'),
            'state': inst.get('state'),
            'min_sz': safe_float(inst.get('minSz')),
            'tick_sz': safe_float(inst.get('tickSz')),
            'lot_sz': safe_float(inst.get('lotSz')),
            'max_lmt_sz': safe_float(inst.get('maxLmtSz')),
            'max_mkt_sz': safe_float(inst.get('maxMktSz')),
            'max_mkt_amt': safe_float(inst.get('maxMktAmt')),
            'ct_mult': safe_float(inst.get('ctMult')),
            'ct_val': safe_float(inst.get('ctVal')),
        }
    
    def set_leverage(self, inst_id: str, lever: int, mgn_mode: str = 'cross', pos_side: str = 'long') -> Dict:
        if lever < 1 or lever > 125:
            return {'status': 'error', 'msg': f"Invalid leverage: {lever}"}
        
        res = self.account_api.set_leverage(lever=str(lever), mgnMode=mgn_mode, instId=inst_id, posSide=pos_side)
        
        if res['code'] == '0' and res['data']:
            data = res['data'][0]
            return {
                'inst_id': data.get('instId'),
                'lever': data.get('lever'),
                'mgn_mode': data.get('mgnMode'),
                'pos_side': data.get('posSide'),
                'status': 'success',
                'msg': 'Leverage set successfully'
            }
        else:
            return {'status': 'error', 'msg': res.get('msg', 'Unknown error'), 'code': res.get('code')}
    
    def convert_coin_contract(self, inst_id: str, sz: float, 
                            convert_type: int = 1, px: Optional[float] = None,
                            unit: str = 'coin') -> Dict:
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
        ord_type = 'limit' if price else 'market'
        client_order_id = _generate_client_order_id()
        
        if self.inst_type == 'SPOT':
            td_mode = 'cash'
        else:
            td_mode = 'cross'
        
        order_params = {
            'instId': inst_id,
            'tdMode': td_mode,
            'side': side,
            'ordType': ord_type,
            'sz': str(size),
            'clOrdId': client_order_id,
            'tag': self.broker_code,
        }
        
        if self.inst_type in ['SWAP', 'FUTURES']:
            if _pos_mode is None:
                account_config = self.get_account_config()
                _pos_mode = account_config.get('pos_mode')
            
            if _pos_mode == 'net_mode':
                order_params['posSide'] = 'net'
            else:
                order_params['posSide'] = pos_side
            order_params['reduceOnly'] = 'true' if reduce_only else 'false'
        
        if not price:
            if self.inst_type == 'SPOT':
                order_params['tgtCcy'] = 'quote_ccy'
        else:
            order_params['px'] = str(price)
        
        res = self.trade_api.place_order(**order_params)
        
        if res['code'] == '0' and res['data']:
            return {
                'order_id': res['data'][0].get('ordId'),
                'client_order_id': client_order_id,
                'status': 'success',
                'msg': 'Order placed successfully'
            }
        else:
            return {
                'status': 'error',
                'msg': res.get('msg', 'Unknown error'),
                'code': res.get('code')
            }
    
    def get_account_config(self) -> Dict:
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

    def rebalance_portfolio(self, target_weights: Dict[str, float], 
                           budget: float = 1000.0,
                           symbol_suffix: str = '-USDT-SWAP',
                           leverage: int = 5) -> Dict:
        if not target_weights:
            return {'status': 'error', 'msg': 'Target weights is empty'}
        
        current_positions = self.get_positions()
        current_holdings = {}
        for pos_key, pos_data in current_positions.items():
            symbol = pos_data['symbol']
            base_symbol = symbol.split('-')[0]
            qty = pos_data['pos_qty']
            notional_usd = pos_data['notional_usd']
            direction = 'long' if notional_usd > 0 else 'short'
            current_holdings[base_symbol] = {
                'inst_id': symbol,
                'side': direction,
                'qty': qty,
                'mark_px': pos_data['mark_px'],
                'entry_px': pos_data['entry_px'],
                'usd_value': notional_usd
            }
        
        total_position_value = sum(abs(h['usd_value']) for h in current_holdings.values())
        base_budget = total_position_value if total_position_value > 0 else budget
        
        min_sizes = {}
        for symbol in target_weights.keys():
            inst_id = f"{symbol}{symbol_suffix}"
            inst_info = self.get_instrument_info(inst_id)
            if 'error' not in inst_info:
                min_sz = inst_info.get('min_sz', 0.01)
                ticker = self.get_ticker(inst_id)
                px = ticker.get('last_px', 0) if 'error' not in ticker else 0
                min_sizes[symbol] = min_sz * px if px > 0 else 0.0
            else:
                min_sizes[symbol] = 0.0
        
        target_holdings = {}
        for symbol, weight in target_weights.items():
            target_usd = weight * base_budget
            target_holdings[symbol] = {
                'weight': weight,
                'target_usd': target_usd,
                'direction': 'long' if weight > 0 else ('short' if weight < 0 else 'none'),
                'min_usd': min_sizes.get(symbol, 0.0)
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
            
            action = 'none'
            order_side = None
            order_size = 0
            
            min_usd_threshold = target.get('min_usd', 0.0) * 1.1
            
            if abs(diff_usd) < max(1.0, min_usd_threshold):
                action = 'none'
            elif current['side'] is None or current['qty'] == 0:
                if target['direction'] == 'long' and diff_usd > 0:
                    action, order_side, order_size = 'long', 'buy', diff_usd
                elif target['direction'] == 'short' and diff_usd < 0:
                    action, order_side, order_size = 'short', 'sell', abs(diff_usd)
            else:
                current_side = current['side']
                target_side = target['direction']
                
                if current_side == target_side:
                    if diff_usd > 0:
                        order_side = 'buy' if current_side == 'long' else 'sell'
                        action = 'long' if current_side == 'long' else 'short'
                        order_size = diff_usd
                    elif diff_usd < 0:
                        action = 'reduce'
                        order_side = 'sell' if current_side == 'long' else 'buy'
                        order_size = abs(diff_usd)
                else:
                    action = 'close'
                    order_side = 'sell' if current_side == 'long' else 'buy'
                    order_size = abs(current['usd_value'])
            
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
            
            if action != 'none' and order_side and order_size > 0:
                if action == 'close' and target['direction'] != 'none':
                    close_record = trade_record.copy()
                    close_record['action'] = 'close'
                    close_record['order_size'] = order_size
                    close_record['msg'] = None
                    orders_to_execute.append({
                        'symbol': symbol,
                        'inst_id': inst_id,
                        'side': order_side,
                        'usd_amount': order_size,
                        'trade_record': close_record,
                        'is_close': True,
                        'is_reverse': True
                    })
                    
                    new_side = 'buy' if target['direction'] == 'short' else 'sell'
                    new_size = abs(target_usd)
                    open_record = {
                        'symbol': symbol,
                        'target_weight': target['weight'],
                        'target_usd': target_usd,
                        'current_usd': 0,
                        'diff_usd': target_usd,
                        'action': 'long' if target['direction'] == 'long' else 'short',
                        'order_size': new_size,
                        'order_id': None,
                        'status': 'pending',
                        'msg': None
                    }
                    orders_to_execute.append({
                        'symbol': symbol,
                        'inst_id': inst_id,
                        'side': new_side,
                        'usd_amount': new_size,
                        'trade_record': open_record,
                        'is_close': False,
                        'is_reverse': True
                    })
                    
                    close_record['is_part_of_reverse'] = True
                    open_record['is_part_of_reverse'] = True
                    rebalance_trades.append(close_record)
                    rebalance_trades.append(open_record)
                else:
                    trade_record['msg'] = None
                    orders_to_execute.append({
                        'symbol': symbol,
                        'inst_id': inst_id,
                        'side': order_side,
                        'usd_amount': order_size,
                        'trade_record': trade_record,
                        'is_close': False,
                        'is_reverse': False
                    })
                    
                    rebalance_trades.append(trade_record)
            else:
                rebalance_trades.append(trade_record)
        
        successful_orders = 0
        failed_orders = 0
        
        trade_records_map = {}
        for trade in rebalance_trades:
            symbol = trade['symbol']
            if symbol not in trade_records_map:
                trade_records_map[symbol] = []
            trade_records_map[symbol].append(trade)
        
        for order_info in orders_to_execute:
            inst_id = order_info['inst_id']
            side = order_info['side']
            usd_amount = order_info['usd_amount']
            trade_record = order_info['trade_record']
            symbol = order_info['symbol']
            
            try:
                inst_info = self.get_instrument_info(inst_id)
                if 'error' not in inst_info:
                    min_sz = inst_info.get('min_sz', 0.01)
                    ticker = self.get_ticker(inst_id)
                    current_px = ticker.get('last_px', 0) if 'error' not in ticker else 0
                    
                    if current_px > 0:
                        min_usd = min_sz * current_px
                        if usd_amount < min_usd:
                            trade_record['status'] = 'skip'
                            trade_record['msg'] = f"USD amount {usd_amount:.2f} < minimum {min_usd:.2f}"
                            continue
                
                lever_result = self.set_leverage(
                    inst_id=inst_id,
                    lever=leverage,
                    mgn_mode='cross',
                    pos_side='long' if side == 'buy' else 'short'
                )
                
                if lever_result['status'] != 'success':
                    trade_record['status'] = 'error'
                    trade_record['msg'] = f"Failed to set leverage: {lever_result.get('msg', 'Unknown error')}"
                    failed_orders += 1
                    continue
                
                ticker = self.get_ticker(inst_id)
                current_px = ticker.get('last_px') if 'error' not in ticker else None
                
                convert_result = self.convert_coin_contract(
                    inst_id=inst_id,
                    sz=usd_amount,
                    convert_type=1,
                    px=current_px,
                    unit='usds'
                )
                
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
                
                action_type = trade_record.get('action')
                reduce_only = action_type in ['reduce', 'close']
                
                order_result = self.place_order(
                    inst_id=inst_id,
                    side=side,
                    size=contract_size,
                    price=None,
                    pos_side='long' if side == 'buy' else 'short',
                    reduce_only=reduce_only
                )
                
                if order_result['status'] == 'success':
                    trade_record['order_id'] = order_result.get('order_id')
                    trade_record['status'] = 'success'
                    successful_orders += 1
                else:
                    trade_record['status'] = 'error'
                    trade_record['msg'] = order_result.get('msg', 'Unknown error')
                    failed_orders += 1
            
            except Exception as e:
                trade_record['status'] = 'error'
                trade_record['msg'] = str(e)
                failed_orders += 1
        
        trades_by_symbol = {}
        for trade in rebalance_trades:
            if trade['symbol'] not in trades_by_symbol:
                trades_by_symbol[trade['symbol']] = []
            trades_by_symbol[trade['symbol']].append(trade)
        
        for symbol, trades in trades_by_symbol.items():
            related_orders = [o for o in orders_to_execute if o['symbol'] == symbol]
            successful_related = sum(1 for o in related_orders if o['trade_record']['status'] == 'success')
            failed_related = sum(1 for o in related_orders if o['trade_record']['status'] == 'error')
            
            if related_orders and related_orders[0].get('is_reverse'):
                if successful_related == 2:
                    for trade in trades:
                        trade['status'] = 'success'
                elif successful_related == 1:
                    for trade in trades:
                        if trade['status'] != 'success':
                            trade['status'] = 'partial'
                elif failed_related > 0:
                    for trade in trades:
                        trade['status'] = 'error'
            else:
                if successful_related > 0 and trades:
                    trades[0]['status'] = 'success'
        
        symbols_rebalanced = sum(1 for t in rebalance_trades if t['status'] == 'success')
        symbols_error = sum(1 for t in rebalance_trades if t['status'] == 'error')
        
        return {
            'status': 'success' if failed_orders == 0 else ('partial' if successful_orders > 0 else 'error'),
            'budget': base_budget,
            'total_position_usd': total_position_value,
            'rebalance_trades': rebalance_trades,
            'summary': {
                'symbols_rebalanced': symbols_rebalanced,
                'symbols_error': symbols_error,
                'total_orders': len(orders_to_execute),
                'successful_orders': successful_orders,
                'failed_orders': failed_orders
            },
            'msg': f'Rebalance completed: {successful_orders} success, {failed_orders} failed'
        }


class Rebalancer:
    """調倉器（高層 API）"""
    
    def __init__(self, target_weights: Dict[str, float], trader: OKXTrader,
                 budget: float = 1000.0, symbol_suffix: str = '-USDT-SWAP',
                 leverage: int = 5):
        self.target_weights = target_weights
        self.trader = trader
        self.budget = budget
        self.symbol_suffix = symbol_suffix
        self.leverage = leverage
        self.result = None
    
    def run(self) -> 'Rebalancer':
        """執行調倉，返回 self 支持鏈式調用"""
        self.result = self.trader.rebalance_portfolio(
            target_weights=self.target_weights,
            budget=self.budget,
            symbol_suffix=self.symbol_suffix,
            leverage=self.leverage
        )
        return self
    
    def summary(self) -> str:
        """生成調倉摘要"""
        if not self.result:
            return "Rebalancer not executed. Call run() first."
        
        lines = ["========== Rebalance Summary =========="]
        lines.append(f"Status: {self.result['status'].upper()}")
        lines.append(f"Budget: ${self.result['budget']:,.2f}")
        lines.append(f"Total Position: ${self.result['total_position_usd']:,.2f}")
        lines.append("")
        
        summary = self.result['summary']
        lines.append("Order Statistics:")
        lines.append(f"  Total: {summary['total_orders']}")
        lines.append(f"  Success: {summary['successful_orders']}")
        lines.append(f"  Failed: {summary['failed_orders']}")
        lines.append("")
        
        lines.append("Rebalance Trades:")
        for trade in self.result['rebalance_trades']:
            if trade['action'] != 'none':
                if trade['status'] == 'success':
                    status_marker = "✓"
                elif trade['status'] == 'partial':
                    status_marker = "⚠️"
                elif trade['status'] == 'skip':
                    status_marker = "⏭️"
                else:
                    status_marker = "✗"
                
                lines.append(
                    f"{status_marker} {trade['symbol']:8} {trade['action']:8} | "
                    f"Target: ${trade['target_usd']:10.2f} | "
                    f"Current: ${trade['current_usd']:10.2f} | "
                    f"Delta: ${trade['diff_usd']:+10.2f}"
                )
        
        return "\n".join(lines)
    
    def print_summary(self) -> 'Rebalancer':
        """打印調倉摘要"""
        print(self.summary())
        return self
    
    def get_result(self) -> Dict:
        """獲取原始結果字典"""
        return self.result or {}
    
    def __repr__(self) -> str:
        if not self.result:
            return (f"Rebalancer(budget={self.budget}, "
                   f"symbols={len(self.target_weights)}, status=not_executed)")
        return (f"Rebalancer(budget={self.budget}, "
               f"status={self.result['status']}, "
               f"success={self.result['summary']['successful_orders']}, "
               f"failed={self.result['summary']['failed_orders']})")


def rebalance(target_weights: Dict[str, float], trader: OKXTrader,
             budget: float = 1000.0, symbol_suffix: str = '-USDT-SWAP',
             leverage: int = 5, auto_run: bool = True) -> Rebalancer:
    """方便調用的調倉函數"""
    rb = Rebalancer(target_weights, trader, budget, symbol_suffix, leverage)
    if auto_run:
        rb.run()
    return rb
