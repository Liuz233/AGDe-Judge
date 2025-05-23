import re


def _parse_output_absolute(output):
    pattern = r"""
        (?:                        # Start of non-capturing group
            \[RESULT\]|\[SCORE\]|   # Match [RESULT] or [SCORE]
            Score:?|score:?|        # Match Score: or score:
            Result:?|\[Result\]:?|  # Match Result: or [Result]:
            score\s+of              # Match "score of"
        )                           # End of non-capturing group
        \s*                         # Allow any whitespace
        (?:\(|\[|\s)*               # Allow opening brackets or whitespace
        (\d+)                       # Capture the digit(s)
        (?:                         # Start of non-capturing group
            (?:\)|\]|\s|$)|         # Allow closing brackets, whitespace, or end of string
            (?:/\s*5|               # Allow /5 with optional whitespace
               \s*out\s*of\s*5)     # or "out of 5" with flexible whitespace
        )?                          # End of non-capturing group
        (?:\s*$)                    # Match from the end of string 
    """
    match = re.search(pattern, output, re.IGNORECASE | re.VERBOSE)

    if match:
        result = int(match.group(1))
        if 1 <= result <= 5:  # Ensure the result is within the valid range
            feedback = output[: match.start()].strip()
            return output, result

    return None, None


def _parse_output_relative(output):
    explicit_pattern = r"""
        (?:                                # Start of non-capturing group
            \[RESULT\]|\[RESULT:\s*|        # Match [RESULT] or [RESULT:
            \[Response\s+|                  # Match [Response
            \[Result\](?:\s+Response)?|     # Match [Result] or [Result] Response
            \[Result:\s*|                   # Match [Result:
            (?:^|\n)Result:?\s*             # Match Result: at the start of a line
        )                                   # End of non-capturing group
        \s*                                 # Allow any whitespace
        (A|B|TIE)                           # Capture A, B, or TIE
        (?:\])?                             # Allow closing bracket, whitespace, or end of string
        (?:\s*$)                            # Match from the end of string 
    """
    match = re.search(
        explicit_pattern, output, re.IGNORECASE | re.VERBOSE | re.MULTILINE
    )

    if match:
        result = match.group(1).upper()
        feedback = output[: match.start()].strip()
        return output, result
    return None, None

def extract_pariwise_result(raw_output):
    raw_tmp = raw_output
    raw_output = raw_output.strip()
    pos = raw_output.rfind('final decision is ')
    pred_label = None
    if pos != -1:
        pred_rest = raw_output[pos + len('final decision is '):].strip().lower()
        if pred_rest.startswith('response 1') or pred_rest.startswith('**response 1'): pred_label = 'A'
        elif pred_rest.startswith('response 2') or pred_rest.startswith('**response 2'): pred_label = 'B'
        elif pred_rest.startswith('tie') or pred_rest.startswith('**tie'): pred_label = 'TIE'
    # return pred_label

    if pred_label is None:
        return None, None
    else:
        return raw_tmp, pred_label

def parse_output(output, mode: str):
    assert mode in [
        "absolute",
        "relative",
        "Auto-J"
    ], "Invalid mode. Supported modes are: 'absolute', 'relative', 'Auto-J'"
    if output is None:
        return None, None
    if mode == "absolute":
        return _parse_output_absolute(output)
    elif mode == "relative":
        return _parse_output_relative(output)
    elif mode == "Auto-J":
        return extract_pariwise_result(output)
