import requests
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import time
import logging
import pytz
from typing import Dict, List
from web3 import Web3

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dai_onchain.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DAIOnChainScraper:
    def __init__(self, db_path='dai_stablecoin.db'):
        self.db_path = db_path
        
        # EVM鏈配置
        self.chains_config = {
            'ethereum': {
                'name': 'Ethereum',
                'rpc': 'https://eth-mainnet.g.alchemy.com/v2/demo',
                'dai_contract': '0x6b175474e89094c44da98b954eedeac495271d0f',
                'block_time': 12,
                'code': 'ETH'
            },
            'arbitrum': {
                'name': 'Arbitrum',
                'rpc': 'https://arb-mainnet.g.alchemy.com/v2/demo',
                'dai_contract': '0xda10009cbd5d07dd0cecc66161fc93d7c9000da1',
                'block_time': 0.25,
                'code': 'ARB'
            },
            'optimism': {
                'name': 'OP Mainnet',
                'rpc': 'https://opt-mainnet.g.alchemy.com/v2/demo',
                'dai_contract': '0xda10009cbd5d07dd0cecc66161fc93d7c9000da1',
                'block_time': 2,
                'code': 'OP'
            },
            'bsc': {
                'name': 'BSC',
                'rpc': 'https://bsc-dataseed.binance.org',
                'dai_contract': '0x1af3f329e8be154074d8769d1ffa4ee058b1dbc3',
                'block_time': 3,
                'code': 'BNB'
            },
            'polygon': {
                'name': 'Polygon',
                'rpc': 'https://polygon-rpc.com',
                'dai_contract': '0x8f3cf7ad23cd3cadbd9735aff958023239c6a063',
                'block_time': 2,
                'code': 'POL'
            },
            'base': {
                'name': 'Base',
                'rpc': 'https://mainnet.base.org',
                'dai_contract': '0x50c5725949a6f0c72e6c4a641f24049a917db0cb',
                'block_time': 2,
                'code': 'BASE'
            }
        }
        
        # Solana配置
        self.solana_config = {
            'name': 'Solana',
            'rpc': 'https://api.mainnet-beta.solana.com',
            'code': 'SOL',
            'dai_mint': None  # 用戶可根據需要配置
        }
        
        # ERC20 ABI
        self.erc20_abi = [
            {
                "constant": True,
                "inputs": [],
                "name": "totalSupply",
                "outputs": [{"name": "", "type": "uint256"}],
                "type": "function"
            },
            {
                "constant": True,
                "inputs": [],
                "name": "decimals",
                "outputs": [{"name": "", "type": "uint8"}],
                "type": "function"
            }
        ]
        
        self.init_database()
    
    def init_database(self):
        """初始化數據庫"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS dai_supply (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                chain TEXT NOT NULL,
                dai_amount REAL NOT NULL,
                block_number INTEGER,
                source TEXT DEFAULT 'onchain_rpc',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(timestamp, chain)
            )
        ''')
        conn.commit()
        conn.close()
        
        import os
        db_path_absolute = os.path.abspath(self.db_path)
        logger.info(f"✓ 數據庫初始化完成")
        logger.info(f"  數據庫位置: {db_path_absolute}")
    
    def get_current_block(self, chain_key: str) -> int:
        """獲取當前區塊號"""
        try:
            config = self.chains_config[chain_key]
            w3 = Web3(Web3.HTTPProvider(config['rpc']))
            
            if not w3.is_connected():
                logger.error(f" 無法連接 {config['name']} RPC")
                return None
            
            block_num = w3.eth.block_number
            logger.debug(f"  {config['name']}: 當前區塊號 {block_num}")
            return block_num
        
        except Exception as e:
            logger.error(f"獲取 {chain_key} 當前區塊失敗: {e}")
            return None
    
    def estimate_block_number(self, chain_key: str, target_timestamp: int, current_block: int) -> int:
        """根據時間戳估算對應的區塊號"""
        try:
            config = self.chains_config[chain_key]
            w3 = Web3(Web3.HTTPProvider(config['rpc']))
            
            current_block_data = w3.eth.get_block(current_block)
            current_timestamp = current_block_data['timestamp']
            
            time_diff = current_timestamp - target_timestamp
            block_diff = int(time_diff / config['block_time'])
            estimated_block = current_block - block_diff
            
            estimated_block = max(0, estimated_block)
            return estimated_block
        
        except Exception as e:
            logger.warning(f"估算 {chain_key} 區塊號失敗: {e}")
            return None
    
    def query_total_supply(self, chain_key: str, block_number: int) -> float:
        """查詢特定區塊的DAI totalSupply"""
        try:
            config = self.chains_config[chain_key]
            w3 = Web3(Web3.HTTPProvider(config['rpc']))
            
            if not w3.is_connected():
                return None
            
            contract_address = Web3.to_checksum_address(config['dai_contract'])
            contract = w3.eth.contract(
                address=contract_address,
                abi=self.erc20_abi
            )
            
            total_supply_raw = contract.functions.totalSupply().call(block_identifier=block_number)
            decimals = contract.functions.decimals().call(block_identifier=block_number)
            total_supply = total_supply_raw / (10 ** decimals)
            
            return total_supply
        
        except Exception as e:
            logger.warning(f"查詢 {chain_key} 區塊 {block_number} 的totalSupply失敗: {e}")
            return None
    
    def fetch_historical_data(self, chain_key: str, days: int = 730) -> List[Dict]:
        """爬取指定鏈過去N天的DAI歷史"""
        if chain_key == 'solana':
            return self.fetch_solana_historical(days)
        
        config = self.chains_config[chain_key]
        logger.info(f"\n【{config['name']}】爬取過去{days}天的歷史數據...")
        
        current_block = self.get_current_block(chain_key)
        if current_block is None:
            logger.error(f" {config['name']} 無法獲取當前區塊")
            return []
        
        end_date = datetime.now(pytz.UTC)
        all_data = []
        failed_days = []
        
        for day_offset in range(days):
            target_date = end_date - timedelta(days=day_offset)
            target_timestamp = int(target_date.timestamp())
            
            estimated_block = self.estimate_block_number(
                chain_key, target_timestamp, current_block
            )
            
            if estimated_block is None:
                failed_days.append(day_offset)
                continue
            
            total_supply = None
            max_retries = 3
            for retry in range(max_retries):
                try:
                    total_supply = self.query_total_supply(chain_key, estimated_block)
                    if total_supply is not None:
                        break
                except Exception as e:
                    if retry < max_retries - 1:
                        wait_time = 2 ** retry
                        logger.debug(f"  {config['name']} 重試 ({retry+1}/{max_retries})，等待{wait_time}秒...")
                        time.sleep(wait_time)
                    else:
                        failed_days.append(day_offset)
            
            if total_supply is not None:
                all_data.append({
                    'timestamp': target_date,
                    'chain': config['code'],
                    'dai_amount': total_supply,
                    'block_number': estimated_block,
                    'source': 'onchain_rpc'
                })
                
                if day_offset % 30 == 0:
                    logger.info(f"  {target_date.date()}: {total_supply:,.0f} DAI @ 區塊 {estimated_block}")
            
            if day_offset % 20 == 0 and day_offset > 0:
                logger.debug(f"  {config['name']} 進度 {day_offset}/{days}，稍等片刻...")
                time.sleep(2)
        
        success_rate = (len(all_data) / days) * 100 if days > 0 else 0
        logger.info(f"✓ {config['name']} 獲取到 {len(all_data)}/{days} 筆數據 ({success_rate:.1f}%)")
        
        if failed_days:
            logger.warning(f"   {config['name']} 失敗了 {len(failed_days)} 天（API限流或超時）")
        
        return all_data
    
    def fetch_solana_historical(self, days: int = 730) -> List[Dict]:
        """查詢Solana上DAI的歷史"""
        if not self.solana_config.get('dai_mint'):
            logger.warning(" Solana: 未配置DAI mint address")
            logger.info("   提示：Solana上DAI流量極少。如果知道官方mint address，請配置在代碼中")
            return []
        
        logger.info(f"\n【Solana】爬取過去{days}天的DAI供應量...")
        
        all_data = []
        end_date = datetime.now(pytz.UTC)
        
        for day_offset in range(days):
            target_date = end_date - timedelta(days=day_offset)
            
            try:
                if day_offset == 0:
                    supply = self.query_solana_dai_supply()
                    if supply is not None:
                        all_data.append({
                            'timestamp': target_date,
                            'chain': self.solana_config['code'],
                            'dai_amount': supply,
                            'block_number': None,
                            'source': 'solana_rpc'
                        })
                        logger.info(f"  {target_date.date()}: {supply:,.0f} DAI")
                    else:
                        logger.warning(f"  Solana: 無法獲取DAI供應量（可能未部署或mint address錯誤）")
                        break
            except Exception as e:
                logger.error(f"  Solana查詢失敗: {e}")
                break
        
        if all_data:
            logger.info(f" Solana: 獲取到 {len(all_data)} 筆數據")
            logger.info("  注意：Solana上DAI歷史數據需要特殊方法，這裡只顯示當前值")
        
        return all_data
    
    def query_solana_dai_supply(self) -> float:
        """查詢Solana上DAI的當前供應量"""
        try:
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getTokenSupply",
                "params": [self.solana_config['dai_mint']]
            }
            
            response = requests.post(
                self.solana_config['rpc'],
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            
            if 'result' in data and data['result']:
                supply_raw = int(data['result']['value']['amount'])
                decimals = int(data['result']['value']['decimals'])
                total_supply = supply_raw / (10 ** decimals)
                return total_supply
            else:
                logger.warning(f"  Solana: {data.get('error', '無返回數據')}")
                return None
        
        except Exception as e:
            logger.error(f"  Solana查詢異常: {e}")
            return None
    
    def fetch_current_data(self) -> List[Dict]:
        """獲取所有鏈的當前DAI供應量"""
        logger.info(f"\n【實時更新】查詢所有鏈的當前DAI供應量...")
        all_data = []
        
        # EVM鏈
        for chain_key, config in self.chains_config.items():
            try:
                current_block = self.get_current_block(chain_key)
                if current_block is None:
                    continue
                
                total_supply = self.query_total_supply(chain_key, current_block)
                
                if total_supply is not None:
                    all_data.append({
                        'timestamp': datetime.now(pytz.UTC),
                        'chain': config['code'],
                        'dai_amount': total_supply,
                        'block_number': current_block,
                        'source': 'onchain_rpc'
                    })
                    
                    logger.info(f"  {config['code']:5}: {total_supply:>20,.0f} DAI @ 區塊 {current_block}")
                
                time.sleep(0.5)
            
            except Exception as e:
                logger.error(f"查詢 {chain_key} 失敗: {e}")
                continue
        
        # Solana
        try:
            if self.solana_config.get('dai_mint'):
                logger.info(f"  查詢 Solana...")
                supply = self.query_solana_dai_supply()
                if supply is not None:
                    all_data.append({
                        'timestamp': datetime.now(pytz.UTC),
                        'chain': self.solana_config['code'],
                        'dai_amount': supply,
                        'block_number': None,
                        'source': 'solana_rpc'
                    })
                    logger.info(f"  SOL : {supply:>20,.0f} DAI")
                else:
                    logger.warning(f"  SOL : 無法獲取（mint address可能錯誤或DAI未部署）")
            else:
                logger.warning(f"  SOL : 未配置DAI mint address")
            time.sleep(0.5)
        except Exception as e:
            logger.warning(f"Solana查詢失敗: {e}")
        
        return all_data
    
    def save_to_database(self, data_list: List[Dict]) -> int:
        """保存數據到數據庫"""
        if not data_list:
            logger.warning("沒有數據可保存")
            return 0
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        saved_count = 0
        
        try:
            for record in data_list:
                cursor.execute('''
                    INSERT OR IGNORE INTO dai_supply 
                    (timestamp, chain, dai_amount, block_number, source)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    record['timestamp'].isoformat(),
                    record['chain'],
                    record['dai_amount'],
                    record.get('block_number'),
                    record.get('source', 'onchain_rpc')
                ))
                saved_count += cursor.rowcount
            
            conn.commit()
            logger.info(f"✓ 保存 {saved_count} 筆新數據到數據庫")
            return saved_count
        
        except Exception as e:
            logger.error(f"保存數據失敗: {e}")
            conn.rollback()
        finally:
            conn.close()
        
        return saved_count
    
    def get_dataframe(self, start_date=None, end_date=None, chain=None):
        """讀取數據為DataFrame"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = "SELECT timestamp, chain, dai_amount, block_number FROM dai_supply WHERE 1=1"
            params = []
            
            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date.isoformat() if isinstance(start_date, datetime) else start_date)
            
            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date.isoformat() if isinstance(end_date, datetime) else end_date)
            
            if chain:
                query += " AND chain = ?"
                params.append(chain)
            
            query += " ORDER BY timestamp ASC"
            
            df = pd.read_sql_query(query, conn, params=params)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            conn.close()
            
            return df
        
        except Exception as e:
            logger.error(f"讀取數據失敗: {e}")
            return pd.DataFrame()
    
    def run_full_scrape(self, days=730):
        """完整爬取所有鏈的歷史數據"""
        print("\n" + "="*70)
        print("DAI 鏈上RPC爬蟲 - 完整爬取（EVM鏈 + Solana）")
        print("="*70)
        
        all_data = []
        
        print(f"\n【階段1】爬取EVM鏈歷史數據({days}天)...")
        for chain_key in list(self.chains_config.keys()):
            chain_data = self.fetch_historical_data(chain_key, days=days)
            all_data.extend(chain_data)
            time.sleep(2)
        
        print(f"\n【階段2】爬取Solana歷史數據({days}天)...")
        solana_data = self.fetch_historical_data('solana', days=days)
        all_data.extend(solana_data)
        time.sleep(2)
        
        print(f"\n【階段3】獲取所有鏈當前快照...")
        current_data = self.fetch_current_data()
        all_data.extend(current_data)
        
        if all_data:
            self.save_to_database(all_data)
        
        print("\n" + "="*70)
        print("爬取完成 - 數據統計")
        print("="*70)
        
        df = self.get_dataframe()
        if not df.empty:
            print(f"\n✓ 數據庫共 {len(df)} 筆記錄")
            print(f"  時間範圍: {df['timestamp'].min()} 至 {df['timestamp'].max()}")
            print(f"\n按鏈統計:")
            print(df['chain'].value_counts().to_string())
            
            print(f"\n最新快照:")
            latest = df[df['timestamp'] == df['timestamp'].max()]
            for _, row in latest.iterrows():
                print(f"  {row['chain']:5}: {row['dai_amount']:>20,.0f} DAI")
        
        print("\n" + "="*70)

if __name__ == "__main__":
    import os
    
    db_path = 'dai_stablecoin.db'
    if os.path.exists(db_path):
        os.remove(db_path)
        logger.info(f"✓ 已刪除舊數據庫: {db_path}")
    
    scraper = DAIOnChainScraper()
    
    print("\n" + "="*70)
    print("DAI 鏈上數據爬蟲 - 選擇運行模式")
    print("="*70)
    print("\n【模式1】爬取2年歷史數據")
    print("【模式2】僅獲取當前快照")
    
    SCRAPE_HISTORY = True  # ← 改為 False 只獲取當前快照
    
    if SCRAPE_HISTORY:
        print("\n【選擇】模式1 - 爬取2年歷史（耗時較長，但完整）")
        print("開始爬取... 請耐心等待（1-2小時）")
        scraper.run_full_scrape(days=730)
    else:
        print("\n【選擇】模式2 - 僅獲取當前快照（快速）")
        current_data = scraper.fetch_current_data()
        scraper.save_to_database(current_data)
    
    print("\n" + "="*70)
    print("數據統計")
    print("="*70)
    df = scraper.get_dataframe()
    if not df.empty:
        print(f"\n✓ 共 {len(df)} 筆記錄")
        print(f"  時間範圍: {df['timestamp'].min()} 至 {df['timestamp'].max()}")
        print(f"\n按鏈統計:")
        print(df['chain'].value_counts().to_string())
        print(f"\n最新快照:")
        latest = df[df['timestamp'] == df['timestamp'].max()]
        for _, row in latest.iterrows():
            print(f"  {row['chain']:5}: {row['dai_amount']:>20,.0f} DAI")
    
    print("\n" + "="*70)
