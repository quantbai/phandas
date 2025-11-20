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
            'dai_mint': 'FYpdBuyAHSbdaAyD1sKkxyLWbAP8uUW9h6uvdhK74ij1'
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
    
    def get_current_block(self, chain_key: str, max_retries: int = 3) -> int:
        """獲取當前區塊號（帶重試機制）"""
        config = self.chains_config[chain_key]
        
        for attempt in range(max_retries):
            try:
                w3 = Web3(Web3.HTTPProvider(config['rpc'], request_kwargs={"timeout": 10}))
                
                if not w3.is_connected():
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        logger.warning(f"⚠️  {config['name']} 連接失敗，{wait_time}秒後重試... ({attempt+1}/{max_retries})")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"❌ 無法連接 {config['name']} RPC (已重試{max_retries}次)")
                        return None
                
                block_num = w3.eth.block_number
                logger.debug(f"  {config['name']}: 當前區塊號 {block_num}")
                return block_num
            
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"⚠️  {config['name']} 查詢區塊失敗，{wait_time}秒後重試... ({attempt+1}/{max_retries})")
                    logger.debug(f"    錯誤: {e}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"❌ {config['name']} 無法獲取區塊 (已重試{max_retries}次)")
                    return None
        
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
    
    def query_total_supply(self, chain_key: str, block_number: int, max_retries: int = 3) -> float:
        """查詢特定區塊的DAI totalSupply（帶重試機制）"""
        config = self.chains_config[chain_key]
        
        for attempt in range(max_retries):
            try:
                w3 = Web3(Web3.HTTPProvider(config['rpc'], request_kwargs={"timeout": 10}))
                
                if not w3.is_connected():
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        logger.debug(f"  {config['name']} 連接失敗，{wait_time}秒後重試...")
                        time.sleep(wait_time)
                        continue
                    else:
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
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.debug(f"  {config['name']} 查詢失敗，{wait_time}秒後重試... ({attempt+1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    logger.warning(f"查詢 {chain_key} 區塊 {block_number} 的totalSupply失敗 (已重試{max_retries}次)")
                    return None
        
        return None
    
    def fetch_historical_data(self, chain_key: str, days: int = 730) -> List[Dict]:
        """每天更新一次DAI供應量快照"""
        if chain_key == 'solana':
            return self.fetch_solana_snapshot()
        
        config = self.chains_config[chain_key]
        logger.info(f"\n【{config['name']}】查詢當前DAI供應量...")
        
        all_data = []
        
        try:
            current_block = self.get_current_block(chain_key)
            if current_block is None:
                logger.error(f"❌ {config['name']} 無法獲取當前區塊")
                return []
            
            total_supply = self.query_total_supply(chain_key, current_block)
            
            if total_supply is not None:
                all_data.append({
                    'timestamp': datetime.now(pytz.UTC),
                    'chain': config['code'],
                    'dai_amount': total_supply,
                    'block_number': current_block,
                    'source': 'onchain_rpc'
                })
                
                logger.info(f"  {config['code']}: {total_supply:,.0f} DAI @ 區塊 {current_block}")
            else:
                logger.warning(f"  {config['code']}: 查詢失敗")
            
        except Exception as e:
            logger.error(f"查詢 {chain_key} 失敗: {e}")
        
        return all_data
    
    def fetch_historical_data_init(self, chain_key: str, days: int = 730, interval: int = 1) -> List[Dict]:
        """【首次初始化用】爬取過去N天的DAI歷史（可調整間隔）"""
        if chain_key == 'solana':
            return self.fetch_solana_snapshot()
        
        config = self.chains_config[chain_key]
        logger.info(f"\n【{config['name']}】爬取過去{days}天的歷史數據（間隔{interval}天）...")
        
        current_block = self.get_current_block(chain_key)
        if current_block is None:
            logger.error(f"❌ {config['name']} 無法獲取當前區塊")
            return []
        
        end_date = datetime.now(pytz.UTC)
        all_data = []
        failed_days = []
        
        day_offsets = list(range(0, days, interval))
        day_offsets.append(days - 1)
        day_offsets = sorted(set(day_offsets), reverse=True)
        
        for idx, day_offset in enumerate(day_offsets):
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
                        logger.debug(f"  {config['name']} 重試 ({retry+1}/{max_retries})...")
                        time.sleep(wait_time)
                    else:
                        failed_days.append(day_offset)
            
            if total_supply is not None:
                all_data.append({
                    'timestamp': target_date,
                    'chain': config['code'],
                    'dai_amount': total_supply,
                    'block_number': estimated_block,
                    'source': 'onchain_rpc_historical'
                })
                
                logger.info(f"  {target_date.date()}: {total_supply:,.0f} DAI")
            
            if (idx + 1) % 5 == 0:
                logger.info(f"  進度: {idx + 1}/{len(day_offsets)}")
                time.sleep(1)
        
        success_count = len(all_data)
        logger.info(f"✓ {config['name']} 獲取到 {success_count} 個時間點的數據")
        
        if failed_days:
            logger.warning(f"  ⚠️ 失敗了 {len(failed_days)} 個時間點")
        
        return all_data
    
    def fetch_solana_snapshot(self) -> List[Dict]:
        """查詢Solana當前DAI供應量快照"""
        if not self.solana_config.get('dai_mint'):
            logger.warning("❌ Solana: 未配置DAI mint address")
            logger.info("   提示：如果知道官方mint address，請告訴我或在代碼中配置")
            return []
        
        logger.info(f"\n【Solana】查詢當前DAI供應量...")
        
        all_data = []
        
        try:
            supply = self.query_solana_dai_supply()
            if supply is not None:
                all_data.append({
                    'timestamp': datetime.now(pytz.UTC),
                    'chain': self.solana_config['code'],
                    'dai_amount': supply,
                    'block_number': None,
                    'source': 'solana_rpc'
                })
                logger.info(f"  SOL: {supply:,.0f} DAI")
            else:
                logger.warning(f"  SOL: 無法獲取DAI供應量（mint address可能錯誤）")
        except Exception as e:
            logger.error(f"  Solana查詢失敗: {e}")
        
        return all_data
    
    def query_solana_dai_supply(self, max_retries: int = 3) -> float:
        """查詢Solana上DAI的當前供應量（帶重試機制）"""
        for attempt in range(max_retries):
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
                    error_msg = data.get('error', '無返回數據')
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        logger.debug(f"  Solana 返回錯誤，{wait_time}秒後重試... ({attempt+1}/{max_retries})")
                        time.sleep(wait_time)
                    else:
                        logger.warning(f"  Solana: {error_msg} (已重試{max_retries}次)")
                    continue
            
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.debug(f"  Solana 超時，{wait_time}秒後重試... ({attempt+1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    logger.error(f"  Solana查詢超時 (已重試{max_retries}次)")
            
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.debug(f"  Solana 查詢異常，{wait_time}秒後重試... ({attempt+1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    logger.error(f"  Solana查詢異常 (已重試{max_retries}次): {e}")
        
        return None
    
    def fetch_current_data(self) -> List[Dict]:
        """獲取所有鏈的當前DAI供應量"""
        logger.info(f"\n【實時更新】查詢所有鏈的當前DAI供應量...")
        all_data = []
        
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
        """每日快照更新"""
        print("\n" + "="*70)
        print("DAI 每日快照更新（EVM鏈 + Solana）")
        print("="*70)
        
        all_data = []
        
        print(f"\n【正在更新】查詢所有鏈當前DAI供應量...")
        
        for chain_key in list(self.chains_config.keys()):
            chain_data = self.fetch_historical_data(chain_key, days=days)
            all_data.extend(chain_data)
            time.sleep(0.5)
        
        solana_data = self.fetch_historical_data('solana', days=days)
        all_data.extend(solana_data)
        time.sleep(0.5)
        
        if all_data:
            self.save_to_database(all_data)
        
        print("\n" + "="*70)
        print("更新完成 - 數據統計")
        print("="*70)
        
        df = self.get_dataframe()
        if not df.empty:
            print(f"\n✓ 數據庫共 {len(df)} 筆記錄")
            print(f"  時間範圍: {df['timestamp'].min()} 至 {df['timestamp'].max()}")
            print(f"  數據覆蓋: 約 {(df['timestamp'].max() - df['timestamp'].min()).days} 天")
            print(f"\n按鏈統計:")
            print(df['chain'].value_counts().to_string())
            
            print(f"\n最新快照:")
            latest = df[df['timestamp'] == df['timestamp'].max()]
            for _, row in latest.iterrows():
                print(f"  {row['chain']:5}: {row['dai_amount']:>20,.0f} DAI")
        
        print("\n" + "="*70)
    
    def run_initial_scrape(self, days=730):
        """【首次初始化用】爬取過去2年的歷史數據"""
        print("\n" + "="*70)
        print("DAI 歷史數據初始化（首次運行用）")
        print("="*70)
        print(f"\n爬取過去 {days} 天（{days//365} 年）的歷史數據...")
        print("每10天查詢一次（加快速度）")
        
        all_data = []
        
        for chain_key in list(self.chains_config.keys()):
            chain_data = self.fetch_historical_data_init(chain_key, days=days)
            all_data.extend(chain_data)
            time.sleep(2)
        
        solana_data = self.fetch_historical_data('solana', days=days)
        all_data.extend(solana_data)
        time.sleep(2)
        
        if all_data:
            self.save_to_database(all_data)
        
        print("\n" + "="*70)
        print("初始化完成 - 數據統計")
        print("="*70)
        
        df = self.get_dataframe()
        if not df.empty:
            print(f"\n✓ 數據庫共 {len(df)} 筆記錄")
            print(f"  時間範圍: {df['timestamp'].min()} 至 {df['timestamp'].max()}")
            print(f"  數據覆蓋: 約 {(df['timestamp'].max() - df['timestamp'].min()).days} 天")
            print(f"\n按鏈統計:")
            print(df['chain'].value_counts().to_string())
            
            print(f"\n最新快照:")
            latest = df[df['timestamp'] == df['timestamp'].max()]
            for _, row in latest.iterrows():
                print(f"  {row['chain']:5}: {row['dai_amount']:>20,.0f} DAI")
        
        print("\n" + "="*70)
        print("\n✅ 初始化完成！")
        print("之後每天運行此腳本時，改為調用 scraper.run_full_scrape()")
        print("="*70)


if __name__ == "__main__":
    import os
    
    db_path = 'dai_stablecoin.db'
    scraper = DAIOnChainScraper()
    
    # 檢查是否是首次運行
    is_first_run = not os.path.exists(db_path)
    
    print("\n" + "="*70)
    print("DAI 每日快照爬蟲 - 啟動")
    print("="*70)
    
    if is_first_run:
        # 首次運行時提示
        print("\n🔍 檢測結果: 首次運行（數據庫不存在）")
        print("\n將執行【歷史數據初始化】補充過去2年的數據")
        print("預計耗時: 2-3 小時")
        print("💡 提示: 中途可按 Ctrl+C 暫停，下次運行會自動恢復")
        print("\n" + "="*70)
        
        confirm = input("\n確認開始初始化? (yes/no): ").strip().lower()
        if confirm != 'yes':
            print("❌ 已取消初始化，退出程式")
            exit(0)
        
        scraper.run_initial_scrape(days=730)
        
        print("\n" + "="*70)
        print("✅ 初始化完成！")
        print("\n之後運行此腳本將自動進入【每日更新模式】")
        print("每天運行一次即可自動累積歷史數據")
        print("="*70)
    
    else:
        # 非首次運行：檢查是否需要補充歷史數據
        df = scraper.get_dataframe()
        record_count = len(df)
        
        print(f"\n🔍 檢測到既有數據庫")
        print(f"  當前記錄數: {record_count} 筆")
        
        need_init = False
        if df.empty:
            print(f"  ⚠️  數據庫為空（可能初始化中斷）")
            need_init = True
        else:
            days_covered = (df['timestamp'].max() - df['timestamp'].min()).days
            print(f"  數據涵蓋: 約 {days_covered} 天")
            print(f"  時間範圍: {df['timestamp'].min().date()} 至 {df['timestamp'].max().date()}")
            
            # 如果數據少於730天，提示補充
            if days_covered < 700:
                print(f"\n⚠️  現有數據不足2年，建議補充歷史數據")
                need_init = True
        
        print("\n" + "="*70)
        
        if need_init:
            print("\n選項:")
            print("  [1] 補充歷史數據 (初始化模式，耗時2-3小時)")
            print("  [2] 跳過，進入每日更新模式 (快速，幾十秒)")
            print("  [0] 退出")
            
            choice = input("\n請選擇 (0/1/2): ").strip()
            
            if choice == '1':
                confirm = input("確認補充歷史數據? (yes/no): ").strip().lower()
                if confirm == 'yes':
                    scraper.run_initial_scrape(days=730)
                else:
                    print("已取消，進入每日更新模式")
                    scraper.run_full_scrape()
            elif choice == '2':
                scraper.run_full_scrape()
            else:
                print("已退出")
                exit(0)
        
        else:
            # 數據已足夠，直接進入每日更新模式
            print(f"✓ 數據充足，進入【每日更新模式】")
            print(f"  用時: 幾秒到幾十秒")
            print("="*70)
            scraper.run_full_scrape()
