# -*- coding: utf-8 -*-
"""
data_provider/yfinance_fetcher 中台股指数获取逻辑的单元测试

使用 unittest.mock 模拟 yfinance API 响应，覆盖：
- _get_tw_main_indices 台股指数批量获取
- 台股指数 Yahoo Finance 符号映射是否正确
- 部分/全部失败的降级场景
"""
import sys
import os
import unittest
from unittest.mock import MagicMock, call, patch
import pandas as pd

# 在导入 data_provider 前 mock 可能缺失的依赖，避免环境差异导致测试无法运行
if 'fake_useragent' not in sys.modules:
    sys.modules['fake_useragent'] = MagicMock()

# 确保能导入项目模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def _make_mock_hist(close: float, prev_close: float, high: float = None, low: float = None) -> pd.DataFrame:
    """构造模拟的 history DataFrame，包含计算涨跌幅所需字段"""
    high = high if high is not None else close + 100
    low = low if low is not None else close - 100
    return pd.DataFrame({
        'Close': [prev_close, close],
        'Open': [prev_close - 50, close - 30],
        'High': [prev_close + 100, high],
        'Low': [prev_close - 100, low],
        'Volume': [1000000.0, 1200000.0],
    }, index=pd.DatetimeIndex(['2025-02-16', '2025-02-17']))


def _make_mock_yf(hist_df: pd.DataFrame):
    """构造模拟的 yf 模块，Ticker().history() 返回给定 DataFrame"""
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = hist_df
    mock_yf = MagicMock()
    mock_yf.Ticker.return_value = mock_ticker
    return mock_yf


class TestTwIndexSymbolMapping(unittest.TestCase):
    """验证台股指数 Yahoo Finance 符号映射的正确性"""

    def setUp(self):
        from data_provider.yfinance_fetcher import YfinanceFetcher
        self.fetcher = YfinanceFetcher()

    def test_tw_indices_mapping_symbols(self):
        """台股指数应使用正确的 Yahoo Finance 符号"""
        mock_yf = MagicMock()
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()
        mock_yf.Ticker.return_value = mock_ticker

        self.fetcher._get_tw_main_indices(mock_yf)

        ticker_calls = [call.args[0] for call in mock_yf.Ticker.call_args_list]

        self.assertIn('^TWII', ticker_calls, '台灣加權指數应使用 ^TWII')
        self.assertIn('^TWOTC', ticker_calls, '台灣櫃檯指數应使用 ^TWOTC')


class TestGetTwMainIndices(unittest.TestCase):
    """_get_tw_main_indices 台股指数批量获取测试"""

    def setUp(self):
        from data_provider.yfinance_fetcher import YfinanceFetcher
        self.fetcher = YfinanceFetcher()

    def test_returns_list_when_all_succeed(self):
        """全部台股指数取数成功时返回指数列表"""
        mock_hist = _make_mock_hist(close=17000.0, prev_close=16800.0)
        mock_yf = _make_mock_yf(mock_hist)

        with patch.dict('sys.modules', {'yfinance': mock_yf}):
            result = self.fetcher.get_main_indices(region='tw')

        self.assertIsNotNone(result)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)

        codes = {item['code'] for item in result}
        self.assertEqual(codes, {'TAIEX', 'TAIROC'})

        for item in result:
            self.assertIn('code', item)
            self.assertIn('name', item)
            self.assertIn('current', item)
            self.assertIn('change_pct', item)
            self.assertIn('prev_close', item)
            self.assertIn('amplitude', item)

    def test_returns_none_when_all_fail(self):
        """全部台股指数取数失败时返回 None"""
        mock_yf = _make_mock_yf(pd.DataFrame())

        with patch.dict('sys.modules', {'yfinance': mock_yf}):
            result = self.fetcher.get_main_indices(region='tw')

        self.assertIsNone(result)


class TestMarketAnalyzerTw(unittest.TestCase):
    """MarketAnalyzer 台股区域文本提示测试"""

    def test_get_index_hint_returns_tw_hint_for_english(self):
        from src.market_analyzer import MarketAnalyzer

        analyzer = MarketAnalyzer(region='tw')
        with patch.object(analyzer, '_get_review_language', return_value='en'):
            hint = analyzer._get_index_hint()

        self.assertEqual(
            hint,
            'Analyze the key moves in the TAIEX, TWOTC, and other major Taiwan market indices.',
        )

    def test_build_review_prompt_contains_tw_analyst_role_in_chinese(self):
        from src.market_analyzer import MarketAnalyzer, MarketOverview

        analyzer = MarketAnalyzer(region='tw')
        overview = MarketOverview(date='2026-05-21')

        with patch.object(analyzer, '_get_review_language', return_value='zh'):
            prompt = analyzer._build_review_prompt(overview, [])

        self.assertIn('你是一位专业的台股市场分析师', prompt)
        self.assertIn('台灣股市大盘复盘报告', prompt)

    def test_get_strategy_prompt_block_returns_tw_chinese_blueprint(self):
        from src.market_analyzer import MarketAnalyzer

        analyzer = MarketAnalyzer(region='tw')
        with patch.object(analyzer, '_get_review_language', return_value='zh'):
            strategy_block = analyzer._get_strategy_prompt_block()

        self.assertIn('## 策略蓝图：台股市场态势策略', strategy_block)
        self.assertIn('加权指数', strategy_block)


if __name__ == '__main__':
    unittest.main()
