
import time
import string
import random
from typing import Dict, List, Optional

def _generate_client_order_id() -> str:
    timestamp = str(int(time.time() * 1000))
    random_suffix = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
    return f"t{timestamp}{random_suffix}"[:32]


class OKXTrader:
    """OKX 永續合約交易接口"""
    
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
        """交易前驗證帳戶設置"""
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
                'value': '买卖模式 (单向持仓)' if pos_mode == 'net_mode' else '开平仓模式 (双向持仓)',
                'required': 'net_mode',
                'status': 'ok' if pos_mode == 'net_mode' else 'error'
            }
        }
        
        error_msg = []
        warning_msg = []
        
        if checks['account_mode']['status'] == 'error':
            error_msg.append(f"帳戶模式必須為 合約模式(2) 或 跨幣種保證金模式(3)，目前為 {checks['account_mode']['value']}")
        
        if checks['position_mode']['status'] == 'error':
            error_msg.append(f"必須使用 買賣模式(單向持倉)，目前為 {checks['position_mode']['value']}")
        
        self.acct_lv = acct_lv_code
        self.pos_mode = pos_mode
        
        if error_msg:
            return {
                'status': 'error',
                'msg': '; '.join(error_msg),
                'checks': checks
            }
        
        return {
            'status': 'ok' if not warning_msg else 'warning',
            'msg': '; '.join(warning_msg) if warning_msg else 'Account config validated',
            'checks': checks,
            'acct_lv': acct_lv_code,
            'pos_mode': pos_mode
        }
    
    def get_positions(self, inst_id: Optional[str] = None) -> Dict[str, Dict]:
        res = self.account_api.get_positions(self.inst_type, inst_id)
        if res['code'] != '0' or not res['data']:
            return {}
        
        positions = {}
        for pos in res['data']:
            pos_qty = float(pos.get('pos', 0))
            
            # 過濾掉零持倉（已平倉的記錄）
            if pos_qty == 0:
                continue
            
            inst = pos.get('instId')
            pos_side = pos.get('posSide')
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
        
        # 跨幣種保證金模式 (模式 3) 下，需要調整參數
        leverage_params = {
            'instId': inst_id,
            'lever': str(lever),
            'mgnMode': mgn_mode
        }
        
        # 只在 open_close 模式 (雙向持倉) 且逐倉時添加 posSide
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
            'tag': '82eebde453a2BCDE',
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
                           budget: Optional[float] = None,
                           symbol_suffix: str = '-USDT-SWAP',
                           leverage: int = 5) -> Dict:
        # 先驗證帳戶設置
        validation = self.validate_account_config()
        if validation['status'] == 'error':
            return {
                'status': 'error',
                'msg': validation['msg'],
                'validation': validation['checks']
            }
        
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
        is_empty_position = total_position_value == 0
        
        if is_empty_position:
            if budget is None or budget <= 0:
                return {
                    'status': 'error',
                    'msg': 'Empty position detected. Must provide budget parameter when starting from scratch.'
                }
            base_budget = budget
        else:
            base_budget = total_position_value
        
        target_holdings = {}
        for symbol, weight in target_weights.items():
            target_usd = weight * base_budget
            target_holdings[symbol] = {
                'weight': weight,
                'target_usd': target_usd,
                'direction': 'long' if weight > 0 else ('short' if weight < 0 else 'none'),
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
            
            # 簡單邏輯：根據 delta 決定買賣
            if abs(diff_usd) < 0.01:  # 忽略極小誤差
                action = 'skip'
                order_side = None
                order_size = 0
            elif diff_usd > 0:
                # 目標大於現有，買入（增加多倉或減少空倉）
                action = 'long'
                order_side = 'buy'
                order_size = diff_usd
            else:  # diff_usd < 0
                # 目標小於現有，賣出（減少多倉或增加空倉）
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
        
        # 根據訂單執行結果更新交易記錄狀態
        for symbol, trades in trades_by_symbol.items():
            related_orders = [o for o in orders_to_execute if o['symbol'] == symbol]
            if related_orders and related_orders[0]['trade_record']['status'] == 'success':
                for trade in trades:
                    trade['status'] = 'success'
        
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
                 budget: Optional[float] = None, symbol_suffix: str = '-USDT-SWAP',
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
        
        # 區分 budget 來源
        budget = self.result['budget']
        total_position = self.result['total_position_usd']
        
        if total_position == 0:
            lines.append(f"Budget: ${budget:,.2f} (來自指定資金)")
        else:
            lines.append(f"Budget: ${budget:,.2f} (來自原有持倉)")
        
        lines.append(f"Total Position: ${total_position:,.2f}")
        lines.append("")
        
        summary = self.result['summary']
        lines.append("Order Statistics:")
        lines.append(f"  Total: {summary['total_orders']}")
        lines.append(f"  Success: {summary['successful_orders']}")
        lines.append(f"  Failed: {summary['failed_orders']}")
        lines.append("")
        
        lines.append("Rebalance Trades:")
        for trade in self.result['rebalance_trades']:
            status = trade['status']
            if status == 'success':
                status_str = "[OK]"
            elif status == 'partial':
                status_str = "[PARTIAL]"
            elif status == 'skip':
                status_str = "[SKIP]"
            elif status == 'error':
                status_str = "[ERROR]"
            elif trade['action'] == 'none':
                status_str = "[NONE]"
            else:
                status_str = "[?]"
            
            action_display = trade['action'] if trade['action'] != 'none' else 'skip'
            
            lines.append(
                f"{status_str} {trade['symbol']:8} {action_display:8} | "
                f"Target: ${trade['target_usd']:10.2f} | "
                f"Current: ${trade['current_usd']:10.2f} | "
                f"Delta: ${trade['diff_usd']:+10.2f}"
            )
            
            if trade['msg']:
                lines.append(f"         └─ {trade['msg']}")
        
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
             budget: Optional[float] = None, symbol_suffix: str = '-USDT-SWAP',
             leverage: int = 5, auto_run: bool = True) -> Rebalancer:
    """方便調用的調倉函數"""
    rb = Rebalancer(target_weights, trader, budget, symbol_suffix, leverage)
    if auto_run:
        rb.run()
    return rb
