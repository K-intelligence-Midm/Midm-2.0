import re
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import ToolParserManager
from vllm.entrypoints.openai.tool_parsers.granite_20b_fc_tool_parser import Granite20bFCToolParser

@ToolParserManager.register_module("midm-parser")
class MidmToolParser(Granite20bFCToolParser):
    def __init__(self, tokenizer: AnyTokenizer):
        super().__init__(tokenizer)

        self.bot_token = "<tool_call>"
        self.tool_start_token = self.bot_token
        self.tool_call_regex = re.compile(r"<tool_call>\s*")