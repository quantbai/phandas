
import time
import string
import random
from typing import Dict, List, Optional, Tuple

OKX_BROKER_CODE = '82eebde453a2BCDE'

def _generate_client_order_id() -> str:
    """Generate OKX-compliant clOrdId (alphanumeric, 1-32 chars)"""
    timestamp = str(int(time.time() * 1000))
    random_suffix = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
    return f"t{timestamp}{random_suffix}"[:32]


class OKXTrader:
    """OKX perpetual/futures trading interface"""
    
    def __init__(self, api_key: str, secret_key: str, passphrase: str, 
                 use_testnet: bool = True, inst_type: str = 'SWAP',
                 broker_code: Optional[str] = None):
        """
        Initialize trader
        
        Parameters
        ----------
        api_key : str
            OKX API Key
        secret_key : str
            OKX Secret Key
        passphrase : str
            OKX Passphrase
        use_testnet : bool
            Use testnet (default True)
        inst_type : str
            Instrument type: 'SWAP' (perpetual), 'FUTURES' (futures), 'SPOT' (spot)
        broker_code : str, optional
            OKX Broker Code (default: module constant)
        """
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
        
        self.account_api = Account.AccountAPI(
            api_key, secret_key, passphrase, False, self.flag
        )
        self.trade_api = Trade.TradeAPI(
            api_key, secret_key, passphrase, False, self.flag
        )
        self.market_api = MarketData.MarketAPI(
            api_key, secret_key, passphrase, False, self.flag
        )
    
    def get_balance(self, ccy: Optional[str] = None) -> Dict:
        """
        Query account balance
        
        Parameters
        ----------
        ccy : str, optional
            Currency code; if None, query all
            
        Returns
        -------
        dict
            {'total_eq': float, 'available': float, 'balances': {ccy: {...}}}
        """
        res = self.account_api.get_account_balance(ccy)
        if res['code'] != '0' or not res['data']:
            return {'error': res.get('msg', 'Unknown error')}
        
        data = res['data'][0]
        result = {
            'total_eq': float(data.get('totalEq', 0)),
            'available': float(data.get('availBal', 0)),
            'balances': {}
        }
        
        for detail in data.get('details', []):
            ccy_code = detail.get('ccy')
            result['balances'][ccy_code] = {
                'available': float(detail.get('availBal', 0)),
                'frozen': float(detail.get('frozenBal', 0)),
                'balance': float(detail.get('bal', 0))
            }
        
        return result
    
    def get_positions(self, inst_id: Optional[str] = None) -> Dict[str, Dict]:
        """
        Query contract positions
        
        Parameters
        ----------
        inst_id : str, optional
            Instrument ID (e.g., 'BTC-USDT-SWAP'); if None, query all
            
        Returns
        -------
        dict
            Position data keyed by '{symbol}_{side}'
        """
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
    
    def get_positions_summary(self) -> Dict:
        """
        Query position summary (all positions aggregated)
        
        Returns
        -------
        dict
            Summary with total PnL and position list
        """
        positions = self.get_positions()
        
        if not positions:
            return {
                'status': 'success',
                'total_positions': 0,
                'total_unrealized_pnl': 0,
                'positions': [],
                'msg': 'No positions'
            }
        
        total_pnl = 0
        positions_list = []
        
        for key, pos in positions.items():
            pnl = pos['unrealized_pnl']
            total_pnl += pnl
            
            positions_list.append({
                'symbol': pos['symbol'],
                'side': pos['pos_side'],
                'qty': pos['pos_qty'],
                'mark_px': pos['mark_px'],
                'unrealized_pnl': pnl,
                'leverage': pos['leverage']
            })
        
        return {
            'status': 'success',
            'total_positions': len(positions),
            'total_unrealized_pnl': total_pnl,
            'positions': positions_list,
            'msg': f'Total PnL: {total_pnl:+.2f} USD'
        }
    
    def get_ticker(self, inst_id: str) -> Dict:
        """
        Get real-time market data
        
        Parameters
        ----------
        inst_id : str
            Instrument ID (e.g., 'BTC-USDT-SWAP')
            
        Returns
        -------
        dict
            Market ticker data
        """
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
    
    def place_order(self, inst_id: str, side: str, size: float, 
                    price: Optional[float] = None, 
                    pos_side: str = 'long',
                    reduce_only: bool = False,
                    _pos_mode: Optional[str] = None) -> Dict:
        """
        Place order (market or limit)
        
        Parameters
        ----------
        inst_id : str
            Instrument ID
        side : str
            'buy' or 'sell'
        size : float
            Order size (quantity or USDT amount for spot)
        price : float, optional
            Limit price; if None, market order
        pos_side : str
            'long' or 'short' (contracts only)
        reduce_only : bool
            Close-only order
        _pos_mode : str, optional
            Internal: position mode (avoids redundant queries)
            
        Returns
        -------
        dict
            {'order_id': str, 'client_order_id': str, 'status': str, 'msg': str}
        """
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
    
    def cancel_order(self, inst_id: str, order_id: Optional[str] = None,
                     client_order_id: Optional[str] = None) -> Dict:
        """
        Cancel order
        
        Parameters
        ----------
        inst_id : str
            Instrument ID
        order_id : str, optional
            Order ID
        client_order_id : str, optional
            Client order ID
            
        Returns
        -------
        dict
            Cancel result
        """
        res = self.trade_api.cancel_order(inst_id, order_id, client_order_id)
        
        if res['code'] == '0' and res['data']:
            return {
                'order_id': res['data'][0].get('ordId'),
                'status': 'success',
                'msg': 'Order cancelled successfully'
            }
        else:
            return {
                'status': 'error',
                'msg': res.get('msg', 'Unknown error'),
                'code': res.get('code')
            }
    
    def get_order(self, inst_id: str, order_id: Optional[str] = None,
                  client_order_id: Optional[str] = None) -> Dict:
        """
        Query order status
        
        Parameters
        ----------
        inst_id : str
            Instrument ID
        order_id : str, optional
            Order ID
        client_order_id : str, optional
            Client order ID
            
        Returns
        -------
        dict
            Order status and details
        """
        res = self.trade_api.get_order(inst_id, order_id, client_order_id)
        
        if res['code'] != '0' or not res['data']:
            return {'status': 'error', 'msg': res.get('msg', 'Unknown error')}
        
        order = res['data'][0]
        return {
            'order_id': order.get('ordId'),
            'symbol': order.get('instId'),
            'side': order.get('side'),
            'pos_side': order.get('posSide'),
            'state': order.get('state'),
            'size': float(order.get('sz', 0)),
            'filled_size': float(order.get('fillSz', 0)),
            'price': float(order.get('px', 0)) if order.get('px') else None,
            'avg_fill_px': float(order.get('avgPx', 0)) if order.get('avgPx') else None,
            'timestamp': order.get('uTime')
        }
    
    def get_pending_orders(self, inst_id: Optional[str] = None) -> List[Dict]:
        """
        Query unfilled orders
        
        Parameters
        ----------
        inst_id : str, optional
            Instrument ID; if None, query all
            
        Returns
        -------
        list
            Unfilled orders
        """
        res = self.trade_api.get_order_list(self.inst_type, instId=inst_id)
        
        if res['code'] != '0' or not res['data']:
            return []
        
        orders = []
        for order in res['data']:
            orders.append({
                'order_id': order.get('ordId'),
                'symbol': order.get('instId'),
                'side': order.get('side'),
                'pos_side': order.get('posSide'),
                'state': order.get('state'),
                'size': float(order.get('sz', 0)),
                'filled_size': float(order.get('fillSz', 0)),
                'price': float(order.get('px', 0)) if order.get('px') else None,
                'avg_fill_px': float(order.get('avgPx', 0)) if order.get('avgPx') else None
            })
        
        return orders
    
    def get_account_config(self) -> Dict:
        """
        Query account configuration
        
        Returns
        -------
        dict
            Account settings (mode, level, KYC, etc.)
        """
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
    
    def set_position_mode(self, pos_mode: str = 'net_mode') -> Dict:
        """
        Set position mode
        
        Parameters
        ----------
        pos_mode : str
            'long_short_mode' (bidirectional) or 'net_mode' (unidirectional)
            
        Returns
        -------
        dict
            Updated position mode or error
        """
        if pos_mode not in ['long_short_mode', 'net_mode']:
            return {
                'status': 'error',
                'msg': f"Invalid pos_mode: {pos_mode}. Must be 'long_short_mode' or 'net_mode'"
            }
        
        res = self.account_api.set_position_mode(pos_mode)
        
        if res['code'] == '0' and res['data']:
            return {
                'pos_mode': res['data'][0].get('posMode'),
                'status': 'success',
                'msg': 'Position mode set successfully'
            }
        else:
            return {
                'status': 'error',
                'msg': res.get('msg', 'Unknown error'),
                'code': res.get('code')
            }
    
    def get_instrument_info(self, inst_id: str) -> Dict:
        """
        Query instrument specifications
        
        Parameters
        ----------
        inst_id : str
            Instrument ID
            
        Returns
        -------
        dict
            Min/max order size, tick precision, contract multiplier, etc.
        """
        res = self.account_api.get_instruments(instType=self.inst_type, instId=inst_id)
        if res['code'] != '0' or not res['data']:
            return {'error': res.get('msg', 'Unknown error')}
        
        inst = res['data'][0]
        
        def safe_float(val):
            """Safely convert to float, handling empty strings"""
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
    
    def set_leverage(self, inst_id: str, lever: int, 
                     mgn_mode: str = 'cross', pos_side: str = 'long') -> Dict:
        """
        Set leverage
        
        Parameters
        ----------
        inst_id : str
            Instrument ID
        lever : int
            Leverage multiplier (1-125)
        mgn_mode : str
            'isolated' (isolated) or 'cross' (cross margin)
        pos_side : str
            'long' or 'short'
            
        Returns
        -------
        dict
            Updated leverage settings or error
        """
        if lever < 1 or lever > 125:
            return {
                'status': 'error',
                'msg': f"Invalid leverage: {lever}. Must be between 1-125"
            }
        
        res = self.account_api.set_leverage(
            lever=str(lever),
            mgnMode=mgn_mode,
            instId=inst_id,
            posSide=pos_side
        )
        
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
            return {
                'status': 'error',
                'msg': res.get('msg', 'Unknown error'),
                'code': res.get('code')
            }
    
    def get_leverage(self, inst_id: str, mgn_mode: str = 'cross') -> List[Dict]:
        """
        Query leverage settings
        
        Parameters
        ----------
        inst_id : str
            Instrument ID
        mgn_mode : str
            'isolated' or 'cross'
            
        Returns
        -------
        list
            Leverage information per position
        """
        res = self.account_api.get_leverage(instId=inst_id, mgnMode=mgn_mode)
        
        if res['code'] != '0' or not res['data']:
            return []
        
        leverages = []
        for item in res['data']:
            leverages.append({
                'inst_id': item.get('instId'),
                'lever': int(item.get('lever', 1)),
                'mgn_mode': item.get('mgnMode'),
                'pos_side': item.get('posSide')
            })
        
        return leverages
    
    def get_max_avail_size(self, inst_id: str, td_mode: str = 'cross') -> Dict:
        """
        Query max available order size/margin
        
        Parameters
        ----------
        inst_id : str
            Instrument ID
        td_mode : str
            'cross' (cross margin), 'isolated' (isolated), 'cash' (spot)
            
        Returns
        -------
        dict
            Max buy/sell sizes
        """
        res = self.account_api.get_max_avail_size(instId=inst_id, tdMode=td_mode)
        
        if res['code'] != '0' or not res['data']:
            return {'error': res.get('msg', 'Unknown error')}
        
        data = res['data'][0]
        return {
            'inst_id': data.get('instId'),
            'avail_buy': float(data.get('availBuy', 0)),
            'avail_sell': float(data.get('availSell', 0))
        }
    
    def adjustment_margin(self, inst_id: str, pos_side: str, 
                         margin_type: str, amt: float) -> Dict:
        """
        Adjust margin for a position
        
        Parameters
        ----------
        inst_id : str
            Instrument ID
        pos_side : str
            'long', 'short', or 'net'
        margin_type : str
            'add' (increase) or 'reduce' (decrease)
        amt : float
            Adjustment amount
            
        Returns
        -------
        dict
            Adjustment result or error
        """
        if margin_type not in ['add', 'reduce']:
            return {
                'status': 'error',
                'msg': f"Invalid margin_type: {margin_type}. Must be 'add' or 'reduce'"
            }
        
        res = self.account_api.adjustment_margin(
            instId=inst_id,
            posSide=pos_side,
            type=margin_type,
            amt=str(amt)
        )
        
        if res['code'] == '0' and res['data']:
            data = res['data'][0]
            return {
                'inst_id': data.get('instId'),
                'pos_side': data.get('posSide'),
                'amt': float(data.get('amt', 0)),
                'type': data.get('type'),
                'status': 'success',
                'msg': 'Margin adjusted successfully'
            }
        else:
            return {
                'status': 'error',
                'msg': res.get('msg', 'Unknown error'),
                'code': res.get('code')
            }
    
    def usd_to_contract_size(self, inst_id: str, usd_amount: float) -> Dict:
        """
        Convert USD amount to contract quantity
        
        Parameters
        ----------
        inst_id : str
            Instrument ID
        usd_amount : float
            USD amount
            
        Returns
        -------
        dict
            Contract size, actual USD value, and remainder
        """
        inst_info = self.get_instrument_info(inst_id)
        if 'error' in inst_info:
            return {'status': 'error', 'msg': inst_info['error']}
        
        ct_val = inst_info['ct_val']
        lot_sz = inst_info['lot_sz']
        
        if not ct_val or ct_val == 0:
            return {
                'status': 'error',
                'msg': f"Unable to get contract value for {inst_id}"
            }
        
        max_contracts = usd_amount / ct_val
        contract_size = (max_contracts // lot_sz) * lot_sz if lot_sz > 0 else max_contracts
        actual_usd = contract_size * ct_val
        remaining_usd = usd_amount - actual_usd
        
        return {
            'inst_id': inst_id,
            'usd_amount': usd_amount,
            'contract_size': contract_size,
            'actual_usd': actual_usd,
            'ct_val': ct_val,
            'lot_sz': lot_sz,
            'remaining_usd': remaining_usd,
            'status': 'success',
            'msg': f"Convert {usd_amount} USD to {contract_size} contracts ({actual_usd} USD)"
        }
    
    def place_order_with_usd(self, inst_id: str, side: str, usd_amount: float,
                            price: Optional[float] = None,
                            pos_side: str = 'long',
                            reduce_only: bool = False) -> Dict:
        """
        Place order by USD amount (auto-converts to quantity)
        
        Parameters
        ----------
        inst_id : str
            Instrument ID
        side : str
            'buy' or 'sell'
        usd_amount : float
            USD amount (auto-converted to quantity)
        price : float, optional
            Limit price; if None, market order
        pos_side : str
            'long' or 'short'
        reduce_only : bool
            Close-only order
            
        Returns
        -------
        dict
            Order result with conversion info
        """
        convert_result = self.usd_to_contract_size(inst_id, usd_amount)
        if convert_result['status'] != 'success':
            return {'status': 'error', 'msg': convert_result['msg']}
        
        contract_size = convert_result['contract_size']
        
        if contract_size <= 0:
            return {
                'status': 'error',
                'msg': f"USD amount {usd_amount} is too small. Minimum: {convert_result['ct_val']} USD"
            }
        
        account_config = self.get_account_config()
        pos_mode = account_config.get('pos_mode')
        
        effective_pos_side = 'net' if pos_mode == 'net_mode' else pos_side
        
        order_result = self.place_order(
            inst_id=inst_id,
            side=side,
            size=contract_size,
            price=price,
            pos_side=effective_pos_side,
            reduce_only=reduce_only,
            _pos_mode=pos_mode
        )
        
        order_result['contract_size'] = contract_size
        order_result['actual_usd'] = convert_result['actual_usd']
        order_result['remaining_usd'] = convert_result['remaining_usd']
        
        return order_result

    def place_multiple_orders(self, orders: List[Dict]) -> Dict:
        """
        Batch place orders (max 20)
        
        Parameters
        ----------
        orders : list
            Order specs with inst_id, side, size, price (optional)
            
        Returns
        -------
        dict
            Batch result with individual order statuses
        """
        if not orders:
            return {'status': 'error', 'msg': 'Orders list is empty'}
        
        if len(orders) > 20:
            return {
                'status': 'error',
                'msg': f'Maximum 20 orders allowed, got {len(orders)}'
            }
        
        batch_orders = []
        account_config = self.get_account_config()
        pos_mode = account_config.get('pos_mode')
        
        for order in orders:
            inst_id = order.get('inst_id')
            side = order.get('side')
            size = order.get('size')
            price = order.get('price')
            pos_side = order.get('pos_side', 'long')
            reduce_only = order.get('reduce_only', False)
            
            if not all([inst_id, side, size]):
                continue
            
            ord_type = 'limit' if price else 'market'
            client_order_id = _generate_client_order_id()
            
            if self.inst_type == 'SPOT':
                td_mode = 'cash'
            else:
                td_mode = 'cross'
            
            order_param = {
                'instId': inst_id,
                'tdMode': td_mode,
                'side': side,
                'ordType': ord_type,
                'sz': str(size),
                'clOrdId': client_order_id,
            }
            
            if self.inst_type in ['SWAP', 'FUTURES']:
                order_param['posSide'] = 'net' if pos_mode == 'net_mode' else pos_side
                order_param['reduceOnly'] = 'true' if reduce_only else 'false'
            
            if price:
                order_param['px'] = str(price)
            elif self.inst_type == 'SPOT':
                order_param['tgtCcy'] = 'quote_ccy'
            
            batch_orders.append(order_param)
        
        if not batch_orders:
            return {'status': 'error', 'msg': 'No valid orders to submit'}
        
        res = self.trade_api.place_multiple_orders(batch_orders)
        
        results = []
        success_count = 0
        failed_count = 0
        
        if res['code'] == '0' and res['data']:
            for item in res['data']:
                if item.get('sCode') == '0':
                    results.append({
                        'order_id': item.get('ordId'),
                        'client_order_id': item.get('clOrdId'),
                        'status': 'success',
                        'msg': 'Order placed successfully'
                    })
                    success_count += 1
                else:
                    results.append({
                        'client_order_id': item.get('clOrdId'),
                        'status': 'error',
                        'msg': item.get('sMsg', 'Unknown error')
                    })
                    failed_count += 1
        else:
            return {
                'status': 'error',
                'total': len(batch_orders),
                'results': [],
                'msg': res.get('msg', 'Unknown error')
            }
        
        return {
            'status': 'success' if failed_count == 0 else 'partial',
            'total': len(batch_orders),
            'success_count': success_count,
            'failed_count': failed_count,
            'results': results,
            'msg': f'Submitted {len(batch_orders)} orders: {success_count} success, {failed_count} failed'
        }

    def convert_coin_contract(self, inst_id: str, sz: float, 
                            convert_type: int = 1, px: Optional[float] = None,
                            unit: str = 'coin') -> Dict:
        """
        Convert between coin and contract quantity
        
        Parameters
        ----------
        inst_id : str
            Instrument ID
        sz : float
            Quantity to convert
        convert_type : int
            1=coin to contract (default), 2=contract to coin
        px : float, optional
            Order price (required for some conversions)
        unit : str
            'coin' (default) or 'usds'
            
        Returns
        -------
        dict
            Converted quantity and price
        """
        import okx.PublicData as PublicData
        
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
            return {
                'status': 'error',
                'msg': res.get('msg', 'Unknown error')
            }
        
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

    def place_orders_with_budget(self, budget: float, 
                                orders_spec: List[Dict]) -> Dict:
        """
        Place multiple orders with budget allocation
        
        Parameters
        ----------
        budget : float
            Total budget (USDT)
        orders_spec : list
            Order specs with inst_id, side, budget_ratio
            
        Returns
        -------
        dict
            Allocation result per order
        """
        if not orders_spec:
            return {'status': 'error', 'msg': 'No orders specified'}
        
        total_ratio = sum(order.get('budget_ratio', 0) for order in orders_spec)
        if abs(total_ratio - 1.0) > 0.01:
            return {
                'status': 'error',
                'msg': f'Budget ratios must sum to 1.0, got {total_ratio}'
            }
        
        allocated = []
        orders_to_place = []
        
        for spec in orders_spec:
            inst_id = spec.get('inst_id')
            side = spec.get('side')
            ratio = spec.get('budget_ratio')
            
            allocated_usd = budget * ratio
            
            ticker = self.get_ticker(inst_id)
            current_px = ticker.get('last_px') if 'error' not in ticker else None
            
            convert_result = self.convert_coin_contract(
                inst_id=inst_id,
                sz=allocated_usd,
                convert_type=1,
                px=current_px,
                unit='usds'
            )
            
            if convert_result['status'] != 'success':
                allocated.append({
                    'inst_id': inst_id,
                    'side': side,
                    'allocated_usd': allocated_usd,
                    'status': 'error',
                    'msg': convert_result['msg']
                })
                continue
            
            contract_size = convert_result['sz']
            
            allocated.append({
                'inst_id': inst_id,
                'side': side,
                'allocated_usd': allocated_usd,
                'contract_size': contract_size,
                'status': 'pending'
            })
            
            orders_to_place.append({
                'inst_id': inst_id,
                'side': side,
                'size': contract_size
            })
        
        if orders_to_place:
            batch_result = self.place_multiple_orders(orders_to_place)
            
            for i, order_result in enumerate(batch_result.get('results', [])):
                if i < len(allocated):
                    if order_result['status'] == 'success':
                        allocated[i]['order_id'] = order_result['order_id']
                        allocated[i]['status'] = 'success'
                    else:
                        allocated[i]['status'] = 'error'
                        allocated[i]['msg'] = order_result['msg']
        
        success_count = sum(1 for a in allocated if a['status'] == 'success')
        
        return {
            'status': 'success' if success_count == len(allocated) else 'partial',
            'total_budget': budget,
            'total_orders': len(orders_spec),
            'success_count': success_count,
            'allocated': allocated,
            'msg': f'Allocated {budget} USDT: {success_count}/{len(orders_spec)} orders succeeded'
        }

    def rebalance_portfolio(self, target_weights: Dict[str, float], 
                           budget: float = 1000.0,
                           symbol_suffix: str = '-USDT-SWAP',
                           leverage: int = 5) -> Dict:
        """
        Rebalance portfolio to target weights
        
        Core logic:
        1. Get current positions (USD values and directions)
        2. Calculate target USD per symbol from target weights
        3. Compute deltas and generate rebalance orders
        4. Execute orders with set leverage
        
        Parameters
        ----------
        target_weights : dict
            Target weight distribution (e.g., {'BTC': 0.5, 'ETH': -0.3})
            Positive = long, negative = short
        budget : float
            Total budget (USDT), default 1000.0
        symbol_suffix : str
            Instrument suffix (default '-USDT-SWAP')
        leverage : int
            Leverage multiplier (1-125), default 5
        
        Returns
        -------
        dict
            Rebalance summary with trade details and statistics
        """
        if not target_weights:
            return {'status': 'error', 'msg': 'Target weights is empty'}
        
        # Step 1: Get current positions
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
        
        # Step 2: Calculate target holdings
        target_holdings = {}
        for symbol, weight in target_weights.items():
            target_usd = weight * budget
            target_holdings[symbol] = {
                'weight': weight,
                'target_usd': target_usd,
                'direction': 'long' if weight > 0 else ('short' if weight < 0 else 'none')
            }
        
        # Step 3: Calculate rebalance orders
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
            
            # Determine rebalance action
            action = 'none'
            order_side = None
            order_size = 0
            
            if abs(diff_usd) < 1:
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
            
            # Build trade record
            trade_record = {
                'symbol': symbol,
                'target_weight': target['weight'],
                'target_usd': target_usd,
                'current_usd': current_usd,
                'diff_usd': diff_usd,
                'action': action,
                'order_size': order_size,
                'order_id': None,
                'status': 'pending'
            }
            
            if action != 'none' and order_side and order_size > 0:
                orders_to_execute.append({
                    'symbol': symbol,
                    'inst_id': inst_id,
                    'side': order_side,
                    'usd_amount': order_size,
                    'trade_record': trade_record
                })
            
            rebalance_trades.append(trade_record)
        
        # Step 4: Execute rebalance orders
        successful_orders = 0
        failed_orders = 0
        
        for order_info in orders_to_execute:
            inst_id = order_info['inst_id']
            side = order_info['side']
            usd_amount = order_info['usd_amount']
            trade_record = order_info['trade_record']
            
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
                
                order_result = self.place_order(
                    inst_id=inst_id,
                    side=side,
                    size=contract_size,
                    price=None,
                    pos_side='long' if side == 'buy' else 'short',
                    reduce_only=False
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
        
        # Step 5: Generate summary
        symbols_rebalanced = sum(1 for t in rebalance_trades if t['status'] == 'success')
        symbols_error = sum(1 for t in rebalance_trades if t['status'] == 'error')
        
        return {
            'status': 'success' if failed_orders == 0 else ('partial' if successful_orders > 0 else 'error'),
            'budget': budget,
            'total_position_usd': sum(abs(h['usd_value']) for h in current_holdings.values()),
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
    """Rebalance executor (high-level API similar to Backtester)"""
    
    def __init__(self, target_weights: Dict[str, float], trader: OKXTrader,
                 budget: float = 1000.0, symbol_suffix: str = '-USDT-SWAP',
                 leverage: int = 5):
        """
        Initialize rebalancer
        
        Parameters
        ----------
        target_weights : dict
            Target weight distribution (e.g., {'BTC': 0.5, 'ETH': -0.3})
        trader : OKXTrader
            Trader instance
        budget : float
            Total budget (USDT)
        symbol_suffix : str
            Instrument suffix
        leverage : int
            Leverage multiplier
        """
        self.target_weights = target_weights
        self.trader = trader
        self.budget = budget
        self.symbol_suffix = symbol_suffix
        self.leverage = leverage
        self.result = None
    
    def run(self) -> 'Rebalancer':
        """Execute rebalance; returns self for chaining"""
        self.result = self.trader.rebalance_portfolio(
            target_weights=self.target_weights,
            budget=self.budget,
            symbol_suffix=self.symbol_suffix,
            leverage=self.leverage
        )
        return self
    
    def summary(self) -> str:
        """Generate rebalance summary"""
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
                status_marker = "✓" if trade['status'] == 'success' else "✗"
                lines.append(
                    f"{status_marker} {trade['symbol']:8} {trade['action']:8} | "
                    f"Target: ${trade['target_usd']:10.2f} | "
                    f"Current: ${trade['current_usd']:10.2f} | "
                    f"Delta: ${trade['diff_usd']:+10.2f}"
                )
        
        return "\n".join(lines)
    
    def print_summary(self) -> 'Rebalancer':
        """Print rebalance summary; returns self"""
        print(self.summary())
        return self
    
    def get_result(self) -> Dict:
        """Get raw rebalance result dictionary"""
        return self.result or {}
    
    def __repr__(self) -> str:
        """String representation"""
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
    """
    Convenient rebalance function (similar to backtest())
    
    Parameters
    ----------
    target_weights : dict
        Target weight distribution
    trader : OKXTrader
        Trader instance
    budget : float
        Total budget (USDT)
    symbol_suffix : str
        Instrument suffix
    leverage : int
        Leverage multiplier (1-125)
    auto_run : bool
        Auto-execute if True
        
    Returns
    -------
    Rebalancer
        Rebalancer instance
        
    Example
    -------
    >>> rebalancer = rebalance(skewness.to_weights(), trader, budget=100000, leverage=5)
    >>> rebalancer.print_summary()
    """
    rb = Rebalancer(target_weights, trader, budget, symbol_suffix, leverage)
    
    if auto_run:
        rb.run()
    
    return rb
