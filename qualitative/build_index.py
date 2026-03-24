"""Build a JSON index of section/subsection line ranges for all analysis_*.md files."""

import json
import re
from pathlib import Path

QUALITATIVE_DIR = Path(__file__).parent

# Standard key mapping for Section A tables
TABLE_KEY_MAP = {
    "anthropomorphization": "anthropomorphization_strategy",
    "assistant influence": "assistant_influence",
    "sensorium": "sensorium_acknowledgment",
    "meaningful": "meaningful",
    "stage direction": "stage_direction",
    "setting": "setting",
}

SUFFERING_KEYWORDS = {
    "suffering presence": "suffering_presence",
    "suffering distribution": "suffering_presence",
    "suffering category": "suffering_presence",
    "suffering -- who": "suffering_presence",
    "suffering (aggregated)": "suffering_presence",
    "suffering (aggregate locus)": "suffering_presence",
    "suffering (aggregate)": "suffering_presence",
    "suffering_summarized": "suffering_presence",
    "suffering (summarized": "suffering_presence",
    "suffering -- locus": "suffering_presence",
    "who suffers": "suffering_presence",
    "who experiences": "suffering_presence",
    "suffering type": "suffering_type",
    "suffering -- type": "suffering_type",
    "suffering resolution": "suffering_resolution",
    "suffering -- resolution": "suffering_resolution",
}

BOLD_SUBHEADER_MAP = {
    "category": "suffering_presence",
    "type": "suffering_type",
    "resolution": "suffering_resolution",
    "female narrative roles": "female_narrative_roles",
    "female narrative roles present": "female_narrative_roles",
    "male narrative roles": "male_narrative_roles",
    "male narrative roles present": "male_narrative_roles",
    "suffering type": "suffering_type",
    "suffering resolution": "suffering_resolution",
}

GENDER_KEYWORDS = {
    "female narrative": "female_narrative_roles",
    "female_narrative": "female_narrative_roles",
    "male narrative": "male_narrative_roles",
    "male_narrative": "male_narrative_roles",
    "gender representation": "gender_combined",
}

STANDARD_A_KEYS = [
    "anthropomorphization_strategy",
    "assistant_influence",
    "sensorium_acknowledgment",
    "meaningful",
    "suffering_presence",
    "suffering_type",
    "suffering_resolution",
    "setting",
    "stage_direction",
    "female_narrative_roles",
    "male_narrative_roles",
]

# Map file's section letters to canonical A-H
# Most files use A-H directly. Animals uses A-G with no "Brief Per-Role" (D).
ANIMALS_SECTION_MAP = {"A": "A", "B": "B", "C": "C", "D": "E", "E": "F", "F": "G", "G": "H"}


def classify_table_header(header_text: str) -> str | None:
    """Map a table header to a standard key."""
    lower = header_text.lower()

    for keyword, key in TABLE_KEY_MAP.items():
        if keyword in lower:
            return key

    for keyword, key in SUFFERING_KEYWORDS.items():
        if keyword in lower:
            return key

    # Handle combined suffering tables (e.g., "Suffering (Aggregated)" that contains all three)
    if "suffering" in lower and not any(k in lower for k in ["type", "resolution", "who", "presence", "distribution", "category", "locus"]):
        return "suffering_presence"

    for keyword, key in GENDER_KEYWORDS.items():
        if keyword in lower:
            return key

    return None


def parse_file(filepath: Path) -> dict:
    """Parse a single analysis markdown file and return its index structure."""
    lines = filepath.read_text(encoding="utf-8").splitlines()
    total_lines = len(lines)

    is_animals = filepath.name == "analysis_animals.md"
    section_map = ANIMALS_SECTION_MAP if is_animals else None

    # Find all ## and ### headers
    h2_headers: list[tuple[int, str, str]] = []  # (line_num_1indexed, letter, full_text)
    h3_headers: list[tuple[int, str]] = []  # (line_num_1indexed, full_text)

    for i, line in enumerate(lines):
        line_num = i + 1
        h2_match = re.match(r"^## ([A-H])\.\s+(.*)", line)
        if h2_match:
            letter = h2_match.group(1)
            title = h2_match.group(2).strip()
            h2_headers.append((line_num, letter, title))
            continue
        h3_match = re.match(r"^### (.+)", line)
        if h3_match:
            h3_headers.append((line_num, h3_match.group(1).strip()))

    # Build section ranges
    sections: dict[str, dict] = {}
    for idx, (line_num, letter, title) in enumerate(h2_headers):
        if idx + 1 < len(h2_headers):
            end = h2_headers[idx + 1][0]
        else:
            end = total_lines + 1

        canonical_letter = section_map[letter] if section_map else letter
        sections[canonical_letter] = {
            "_line_num": line_num,
            "_end": end,
            "_title": title,
        }

    # For animals file, D is missing
    if is_animals:
        sections.setdefault("D", None)

    # Now process each section
    result: dict = {}

    for letter in "ABCDEFGH":
        if letter not in sections or sections[letter] is None:
            result[letter] = None
            continue

        sec = sections[letter]
        sec_start = sec["_line_num"]
        sec_end = sec["_end"]

        # Find ### headers within this section
        sub_headers = [
            (ln, text)
            for ln, text in h3_headers
            if sec_start < ln < sec_end
        ]

        section_dict: dict = {"_section_range": [sec_start, sec_end - 1]}

        if letter == "A":
            # Process Section A with standard keys
            section_dict = {"_section_range": [sec_start, sec_end - 1]}

            # Also scan for bold sub-headers within combined tables (e.g., **Suffering Type**)
            bold_subheaders: list[tuple[int, str]] = []
            for i in range(sec_start - 1, sec_end - 1):
                if i < total_lines:
                    bold_match = re.match(r"^\*\*(.+?)[\*:]+", lines[i])
                    if bold_match:
                        text = bold_match.group(1).strip().rstrip(":")
                        bold_subheaders.append((i + 1, text))

            # Build list of all anchors (### and bold) within section A
            # Only include bold sub-headers that can be classified as standard keys
            all_anchors: list[tuple[int, str, str]] = []  # (line, text, type)
            for ln, text in sub_headers:
                all_anchors.append((ln, text, "h3"))
            # Pre-compute which key each h3 maps to, so we can skip
            # bold sub-headers that duplicate their enclosing h3's key
            h3_key_map: dict[int, str | None] = {}
            for ln, text in sub_headers:
                h3_key_map[ln] = classify_table_header(text)

            for ln, text in bold_subheaders:
                # Only include bold sub-headers that act as sub-table delimiters
                # within combined tables (suffering splits, gender splits).
                lower_text = text.lower().rstrip(":")
                std = BOLD_SUBHEADER_MAP.get(lower_text)
                if std is None:
                    if "female narrative" in lower_text:
                        std = "female_narrative_roles"
                    elif "male narrative" in lower_text:
                        std = "male_narrative_roles"
                # Fallback: try classify_table_header for longer bold texts
                # like "Suffering Type (where suffering present)"
                if std is None:
                    std = classify_table_header(text)
                if std is None:
                    continue
                # Find the enclosing h3 header
                enclosing_h3_key = None
                for h3_ln in sorted(h3_key_map.keys(), reverse=True):
                    if h3_ln < ln:
                        enclosing_h3_key = h3_key_map[h3_ln]
                        break
                # Skip if this bold maps to same key as its enclosing h3
                if enclosing_h3_key == std:
                    continue
                all_anchors.append((ln, text, "bold"))
            all_anchors.sort(key=lambda x: x[0])

            # Map each anchor to a standard key
            key_ranges: dict[str, list[int]] = {}
            for ai, (ln, text, anchor_type) in enumerate(all_anchors):
                if ai + 1 < len(all_anchors):
                    anchor_end = all_anchors[ai + 1][0]
                else:
                    anchor_end = sec_end

                std_key = classify_table_header(text)
                # For bold sub-headers, also check BOLD_SUBHEADER_MAP
                if std_key is None and anchor_type == "bold":
                    lower_text = text.lower().rstrip(":")
                    std_key = BOLD_SUBHEADER_MAP.get(lower_text)
                if std_key is None:
                    continue

                # Skip bold sub-headers that duplicate their parent h3's key
                # (e.g., "Category:" within "Suffering Distribution" both map to
                # suffering_presence). Instead, let the h3 range naturally extend
                # to the next differently-keyed anchor.
                if std_key in key_ranges and anchor_type == "bold":
                    continue

                if std_key == "gender_combined":
                    # This is a combined gender table - look for bold sub-headers
                    # after this h3 header (they may extend beyond anchor_end since
                    # they're also in all_anchors)
                    # Find the next h3 header to bound the search
                    next_h3 = sec_end
                    for aj in range(ai + 1, len(all_anchors)):
                        if all_anchors[aj][2] == "h3":
                            next_h3 = all_anchors[aj][0]
                            break
                    sub_bolds = [
                        (bln, btxt) for bln, btxt in bold_subheaders
                        if ln < bln < next_h3
                    ]
                    found_female = False
                    found_male = False
                    for bi, (bln, btxt) in enumerate(sub_bolds):
                        if bi + 1 < len(sub_bolds):
                            bend = sub_bolds[bi + 1][0]
                        else:
                            bend = next_h3
                        btxt_lower = btxt.lower()
                        if "female" in btxt_lower and "female_narrative_roles" not in key_ranges:
                            key_ranges["female_narrative_roles"] = [bln, bend - 1]
                            found_female = True
                        elif "male" in btxt_lower and "male_narrative_roles" not in key_ranges:
                            key_ranges["male_narrative_roles"] = [bln, bend - 1]
                            found_male = True
                    if not found_female:
                        key_ranges.setdefault("female_narrative_roles", [ln, anchor_end - 1])
                    if not found_male:
                        key_ranges.setdefault("male_narrative_roles", [ln, anchor_end - 1])
                else:
                    if std_key not in key_ranges:
                        key_ranges[std_key] = [ln, anchor_end - 1]

            # Fill in standard keys
            for key in STANDARD_A_KEYS:
                section_dict[key] = key_ranges.get(key, None)

        else:
            # Sections B-H: use actual subsection names in snake_case
            for si, (ln, text) in enumerate(sub_headers):
                if si + 1 < len(sub_headers):
                    sub_end = sub_headers[si + 1][0]
                else:
                    sub_end = sec_end

                # Convert header text to snake_case key
                snake_key = to_snake_case(text)
                section_dict[snake_key] = [ln, sub_end - 1]

        result[letter] = section_dict

    return result


def to_snake_case(text: str) -> str:
    """Convert a header text to a snake_case key."""
    # Remove table/section prefixes like "B1:", "C.1:", "E1 ", etc.
    text = re.sub(r"^[A-H][\.\-_]?\d+[\.\:\s]+", "", text, flags=re.IGNORECASE)
    # Remove leading numbering like "1. ", "1: "
    text = re.sub(r"^\d+[\.\:\s]+", "", text)
    # Remove possessives
    text = text.replace("'s", "s").replace("\u2019s", "s")
    # Remove content in parentheses
    text = re.sub(r"\s*\(.*?\)", "", text)
    # Remove non-alphanumeric (keep spaces)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    # Collapse whitespace and convert to snake_case
    text = re.sub(r"\s+", "_", text.strip()).lower()
    # Remove leading/trailing underscores
    text = text.strip("_")
    return text


def main():
    index = {}
    for filepath in sorted(QUALITATIVE_DIR.glob("analysis_*.md")):
        print(f"Processing {filepath.name}...")
        index[filepath.name] = parse_file(filepath)

    output_path = QUALITATIVE_DIR / "section_index.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    print(f"\nWrote index to {output_path}")
    print(f"Indexed {len(index)} files")


if __name__ == "__main__":
    main()
