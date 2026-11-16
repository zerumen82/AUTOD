import re, pathlib
p = pathlib.Path("roop/img_editor/img_editor_manager.py")
s = p.read_text(encoding="utf-8")
s = s.replace("from typing import Optional, Tuple, Dict", "from typing import Any, Dict, Optional, Tuple", 1)
s = s.replace('''                            if not (self._last_instruction_compliance or {}).get(
                                "compliant") is False:''', '''                            if (self._last_instruction_compliance or {}).get(
                                "compliant") is not False:''', 1)
s = s.replace('''        patterns = (
            rf"\\b(?:add|adds|adding|place|places|placing|put|puts|insert|inserts|"''', '''        patterns = (
            r"\\b(?:add|adds|adding|place|places|placing|put|puts|insert|inserts|"''', 1)
p.write_text(s, encoding="utf-8")
print("ok")
