"""Turn the failures of a JUnit report into GitHub annotations.

The log of a workflow run can be read only by a signed-in user; annotations are public. This script prints one
``::error`` line with the names of all failed tests and one per failure (the first nine) with the end of its message.

Usage: python .github/annotate_failures.py test-report.xml
"""

import sys
import xml.etree.ElementTree as ET


def _escape(text: str) -> str:
    return text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def main(path: str) -> int:
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as error:
        print(f"::error title=No test report::{_escape(str(error))}")
        return 0
    failures = []
    for case in root.iter("testcase"):
        for kind in ("failure", "error"):
            node = case.find(kind)
            if node is not None:
                name = f"{case.get('classname', '')}::{case.get('name', '')}"
                text = (node.get("message") or "") + "\n" + (node.text or "")
                failures.append((name, text.strip()))
    if not failures:
        print("::notice title=Test report::no failed test in the report")
        return 0
    names = "\n".join(name for name, _ in failures)
    print(f"::error title={len(failures)} failed tests::{_escape(names[:6000])}")
    for name, text in failures[:9]:
        print(f"::error title={_escape(name[-180:])}::{_escape(text[-1500:])}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "test-report.xml"))
