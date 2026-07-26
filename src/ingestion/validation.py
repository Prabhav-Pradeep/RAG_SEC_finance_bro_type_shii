import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def validate_parsed_document(parsed_blocks: list, filepath: str) -> tuple[bool, str]:
    if not parsed_blocks:
        return False, "Validation Failed: Parser returned 0 blocks."

    total_words = 0
    text_block_count = 0
    table_block_count = 0
    total_chars = 0
    non_alpha_chars = 0

    for block in parsed_blocks:
        content = block.get("content", "")
        block_type = block.get("type", "")

        if block_type == "table":
            table_block_count += 1
        elif block_type == "text":
            text_block_count += 1

        words = content.split()
        total_words += len(words)
        total_chars += len(content)

        for char in content:
            if not char.isalnum() and not char.isspace():
                non_alpha_chars += 1

    if total_words < 500:
        return False, f"Validation Failed: Suspiciously low word count ({total_words} words)."

    if table_block_count == 0:
        return False, "Validation Failed: No financial tables were extracted from filing."

    if total_chars > 0:
        garbage_ratio = non_alpha_chars / total_chars
        if garbage_ratio > 0.35:
            return False, f"Validation Failed: High garbage ratio ({garbage_ratio:.2%})."

    logging.info(
        f"Validation PASSED for {filepath} | Blocks: {len(parsed_blocks)} "
        f"(Text: {text_block_count}, Tables: {table_block_count}) | Words: {total_words}"
    )
    return True, "Passed all quality gates."


# --- EXECUTION BLOCK (For Testing) ---
if __name__ == "__main__":
    # Test 1: Mock Good Data
    good_data = [
        {"type": "text", "content": "This is a clean financial section discussing revenue and risk factors. " * 50},
        {"type": "table", "content": "| Year | Revenue |\n| 2023 | $100M |"}
    ]
    is_valid, reason = validate_parsed_document(good_data, "test_good.htm")
    print(f"Good Test -> Valid: {is_valid} | Reason: {reason}")

    # Test 2: Mock Garbage Data
    bad_data = [
        {"type": "text", "content": "%%%%%$$$$$#####@@@@@!!!!!! " * 20}
    ]
    is_valid, reason = validate_parsed_document(bad_data, "test_bad.htm")
    print(f"Bad Test -> Valid: {is_valid} | Reason: {reason}")