import json
import logging

from app.observability.logging import JsonFormatter, log_event
from app.observability.sanitizer import mask_text, sanitize


def test_phone_masked():
    assert mask_text("联系 13812345678 即可") == "联系 138****5678 即可"


def test_id_card_masked_not_as_phone():
    out = mask_text("身份证 11010119900101123X")
    assert out == "身份证 110101********123X"


def test_email_masked():
    assert mask_text("hr.zhang@acme.example") == "h***@acme.example"


def test_api_key_and_bearer_masked():
    assert "abcdefghijkl" not in mask_text("key=sk-ant-abcdefghijklmnop")
    assert "eyJhbGciOi" not in mask_text("Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.x.y")


def test_sensitive_keys_redacted_recursively():
    out = sanitize({"headers": {"Authorization": "Bearer x"}, "list": ["13812345678"]})
    assert out == {"headers": {"Authorization": "[REDACTED]"}, "list": ["138****5678"]}


def test_json_formatter_masks_fields(caplog):
    logger = logging.getLogger("t")
    with caplog.at_level(logging.INFO, logger="t"):
        log_event(logger, "e", query="我的手机 13900001111")
    line = JsonFormatter().format(caplog.records[0])
    assert json.loads(line)["query"] == "我的手机 139****1111"
