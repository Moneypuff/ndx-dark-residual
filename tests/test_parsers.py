"""Tests for the data-ingestion parsers -- the boundary where raw upstream
documents (FINRA short-volume files, iShares/SSGA holdings exports, SEC EDGAR
acceptance timestamps) enter the pipeline.

These functions all swallow malformed input and are expected to degrade
gracefully (return ``[]`` / ``{}`` / a sensible classification) rather than
crash, because the upstream formats drift and occasionally a gate/HTML page is
served instead of the real document. The tests exercise the happy path plus
the documented "no data / not parseable" fallbacks and the fiddly edges
(header detection, empty-cell indexing, share-class folding, blank/trailer
rows, timezone boundaries).
"""
import io

import pandas as pd
import pytest

import ndx_dark_residual as N
import fetch_earnings_edgar as F


# ===========================================================================
# FINRA short-volume file parsing
# ===========================================================================
FINRA_BODY = (
    "Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market\n"
    "20240125|AAPL|1000|5|4000|B,Q,N\n"
    "20240125|MSFT|2500|0|5000|B,Q,N\n"
)


def test_parse_finra_volumes_happy_path():
    out = N._parse_finra_volumes(FINRA_BODY)
    assert out == {"AAPL": (1000.0, 4000.0), "MSFT": (2500.0, 5000.0)}


def test_parse_finra_volumes_skips_header_and_short_lines():
    body = FINRA_BODY + "20240125|BADROW|1\n" + "\n"  # too few fields, and a blank line
    out = N._parse_finra_volumes(body)
    assert set(out) == {"AAPL", "MSFT"}


def test_parse_finra_volumes_skips_nonnumeric_footer():
    # FINRA appends a summary/footer line whose volume fields aren't numbers.
    body = FINRA_BODY + "20240125|Grand Total|N/A|N/A|N/A|\n"
    out = N._parse_finra_volumes(body)
    assert "Grand Total" not in out
    assert set(out) == {"AAPL", "MSFT"}


def test_parse_finra_volumes_empty_body():
    assert N._parse_finra_volumes("") == {}


@pytest.mark.parametrize("status,text,expected", [
    (404, "", True),                                  # CDN 404 = no file that day
    (403, "<Error><Code>AccessDenied</Code></Error>", True),
    (403, "<Error><Code>NoSuchKey</Code></Error>", True),
    (403, "some other forbidden reason", False),      # a real 403, treat as transient
    (200, "ok", False),
    (500, "server error", False),                     # transient outage -> retry, not "no data"
    (429, "throttled", False),
])
def test_finra_missing_classification(status, text, expected):
    assert N._finra_missing(status, text) is expected


# ===========================================================================
# iShares SpreadsheetML (Excel-XML) holdings -- tickers and weights
# ===========================================================================
def _spreadsheetml(rows_xml):
    return (
        '<?xml version="1.0"?>\n'
        '<Workbook xmlns="urn:schemas-microsoft-com:office:spreadsheet"\n'
        '          xmlns:ss="urn:schemas-microsoft-com:office:spreadsheet">\n'
        ' <Worksheet ss:Name="Holdings"><Table>\n' + rows_xml +
        ' </Table></Worksheet>\n</Workbook>'
    )


def _cell(v, typ="String"):
    return f'<Cell><Data ss:Type="{typ}">{v}</Data></Cell>'


HEADER_ROW = "<Row>" + _cell("Ticker") + _cell("Name") + _cell("Weight (%)") + _cell("Asset Class") + "</Row>\n"


def _data_row(tkr, name, weight, asset="Equity"):
    return ("<Row>" + _cell(tkr) + _cell(name) + _cell(weight, "Number") + _cell(asset) + "</Row>\n")


def test_spreadsheetml_tickers_happy_path_equity_only():
    body = _spreadsheetml(
        "<Row>" + _cell("Fund disclaimer A &amp; B") + "</Row>\n"  # preamble, and a raw & escaped
        + HEADER_ROW
        + _data_row("AAPL", "Apple", "7.25")
        + _data_row("MSFT", "Microsoft", "6.50", asset="Cash")     # non-equity dropped
        + _data_row("USD", "Cash", "0.50", asset="Cash")
    )
    assert N._iwm_tickers_from_spreadsheetml(body) == ["AAPL"]


def test_spreadsheetml_deduplicates():
    body = _spreadsheetml(HEADER_ROW
                          + _data_row("AAPL", "Apple", "7.25")
                          + _data_row("AAPL", "Apple dup", "0.0"))
    assert N._iwm_tickers_from_spreadsheetml(body) == ["AAPL"]


def test_spreadsheetml_handles_ss_index_gaps():
    # SpreadsheetML omits empty cells and jumps columns via ss:Index; the parser
    # must track the true column so Ticker/Asset-Class stay aligned.
    rows = ('<Row>' + _cell("Ticker") + '<Cell ss:Index="3">'
            '<Data ss:Type="String">Asset Class</Data></Cell></Row>\n'
            '<Row>' + _cell("NVDA") + '<Cell ss:Index="3">'
            '<Data ss:Type="String">Equity</Data></Cell></Row>\n')
    assert N._iwm_tickers_from_spreadsheetml(_spreadsheetml(rows)) == ["NVDA"]


def test_spreadsheetml_not_xml_returns_empty():
    assert N._iwm_tickers_from_spreadsheetml("<html>gate page</html>") == []
    assert N._iwm_tickers_from_spreadsheetml("plain,csv,body") == []


def test_spreadsheetml_no_ticker_header_returns_empty():
    body = _spreadsheetml("<Row>" + _cell("Symbol") + _cell("Name") + "</Row>\n")
    assert N._iwm_tickers_from_spreadsheetml(body) == []


def test_spreadsheetml_weights_fold_goog_into_googl():
    # GOOG (class C) weight is folded into GOOGL to match the share-class merge.
    body = _spreadsheetml(HEADER_ROW
                          + _data_row("AAPL", "Apple", "7.25")
                          + _data_row("GOOGL", "Alphabet A", "2.10")
                          + _data_row("GOOG", "Alphabet C", "1.90")
                          + _data_row("USD", "Cash", "0.50", asset="Cash"))
    w = N._ishares_weights_from_spreadsheetml(body)
    assert w == {"AAPL": pytest.approx(7.25), "GOOGL": pytest.approx(4.0)}


def test_spreadsheetml_weights_missing_weight_column_returns_empty():
    body = _spreadsheetml("<Row>" + _cell("Ticker") + _cell("Name") + "</Row>\n"
                          + "<Row>" + _cell("AAPL") + _cell("Apple") + "</Row>\n")
    assert N._ishares_weights_from_spreadsheetml(body) == {}


def test_spreadsheetml_weights_not_xml_returns_empty():
    assert N._ishares_weights_from_spreadsheetml("<html/>oops") == {}


# ===========================================================================
# iShares IWM holdings CSV
# ===========================================================================
IWM_CSV = (
    "iShares Russell 2000 ETF\n"
    'Fund Holdings as of,"Jan 25, 2024"\n'
    "\n"
    "Ticker,Name,Asset Class,Weight (%)\n"
    "SUPN,Supernus,Equity,0.05\n"
    "AMC,AMC Entertainment,Equity,0.04\n"
    "USD,US Dollar,Cash,0.10\n"
    "SUPN,Supernus dup,Equity,0.05\n"
    "-,Futures,Cash and/or Derivatives,0.01\n"
)


def test_iwm_csv_happy_path_equity_dedup():
    # Equity only, de-duplicated, cash/derivative rows skipped.
    assert N._iwm_tickers_from_csv_text(IWM_CSV) == ["SUPN", "AMC"]


def test_iwm_csv_no_holdings_table_returns_empty():
    assert N._iwm_tickers_from_csv_text("<html>no table served</html>") == []


def test_iwm_csv_blank_ticker_trailer_row_is_skipped():
    # Regression: a disclaimer/trailer row with an empty ticker cell must be
    # skipped, not crash (str .strip on a NaN float) or emit a bogus "NAN".
    csv = (
        "Ticker,Name,Asset Class,Weight (%)\n"
        "SUPN,Supernus,Equity,0.05\n"
        ",Disclaimer: this fund ...,Equity,0.00\n"
    )
    assert N._iwm_tickers_from_csv_text(csv) == ["SUPN"]


# ===========================================================================
# SSGA Select-Sector holdings .xlsx
# ===========================================================================
def _ssga_xlsx(rows):
    buf = io.BytesIO()
    pd.DataFrame(rows).to_excel(buf, header=False, index=False, engine="openpyxl")
    return buf.getvalue()


SSGA_ROWS = [
    ["Fund Name:", "Technology Select Sector", None, None],
    ["Ticker Symbol:", "XLK", None, None],
    [None, None, None, None],
    ["Name", "Ticker", "Identifier", "Weight"],
    ["Apple Inc", "AAPL", "03783310", "22.5"],
    ["Microsoft", "MSFT", "59491810", "21.0"],
    ["Cash", "-", "-", "0.1"],
    ["Apple dup", "AAPL", "03783310", "0.0"],
]


def test_ssga_xlsx_happy_path_dedup_and_skip_nonticker():
    assert N._ssga_tickers_from_xlsx(_ssga_xlsx(SSGA_ROWS)) == ["AAPL", "MSFT"]


def test_ssga_xlsx_trailer_row_with_blank_ticker_skipped():
    # Regression: a disclaimer trailer (blank ticker cell) must not become "NAN".
    rows = SSGA_ROWS + [["Disclaimer: past performance ...", None, None, None]]
    assert N._ssga_tickers_from_xlsx(_ssga_xlsx(rows)) == ["AAPL", "MSFT"]


def test_ssga_xlsx_not_a_workbook_returns_empty():
    assert N._ssga_tickers_from_xlsx(b"this is not an xlsx file") == []


def test_ssga_xlsx_no_ticker_column_returns_empty():
    rows = [["Name", "Identifier", "Weight"], ["Apple", "03783310", "22.5"]]
    assert N._ssga_tickers_from_xlsx(_ssga_xlsx(rows)) == []


# ===========================================================================
# SEC EDGAR acceptance-timestamp classification
# ===========================================================================
def test_classify_amc_after_close():
    date, et, timing = F.classify("2024-01-25T16:05:30.000000-05:00")
    assert timing == "amc"
    assert date == "2024-01-25"
    assert et == "2024-01-25 16:05 ET"


def test_classify_bmo_before_open():
    assert F.classify("2024-01-25T08:00:00.000000-05:00")[2] == "bmo"


def test_classify_intraday():
    assert F.classify("2024-01-25T12:30:00.000000-05:00")[2] == "intraday"


@pytest.mark.parametrize("hhmmss,expected", [
    ("09:29:59", "bmo"),        # strictly before 09:30
    ("09:30:00", "intraday"),   # 09:30 is not before open
    ("15:59:59", "intraday"),
    ("16:00:00", "amc"),        # 16:00 is at/after close
])
def test_classify_open_close_boundaries(hhmmss, expected):
    assert F.classify(f"2024-01-25T{hhmmss}.000000-05:00")[2] == expected


def test_classify_converts_utc_offset_to_eastern():
    # 21:30 UTC on 2024-01-25 is 16:30 EST the same ET calendar day -> amc.
    date, et, timing = F.classify("2024-01-25T21:30:00.000000+00:00")
    assert (date, timing) == ("2024-01-25", "amc")
    assert et == "2024-01-25 16:30 ET"


def test_classify_keys_announce_date_off_eastern_not_utc():
    # 02:00 UTC on the 26th is 21:00 ET on the 25th; the announce date is the ET
    # day the news actually hit, not the UTC date.
    date, et, timing = F.classify("2024-01-26T02:00:00.000000+00:00")
    assert date == "2024-01-25"
    assert timing == "amc"


def test_classify_respects_daylight_saving_offset():
    # A summer (EDT, -04:00) after-close timestamp still classifies as amc.
    assert F.classify("2024-06-25T16:05:30.000000-04:00")[2] == "amc"
