def filter_upper(s:str):
    return ' '.join([word for word in s.split() if word[0].islower()])
def trim_line(s: str):
    """
        This function trims off all the punctuations in the line and collapse the
        spaces in the text.
        * The return string is going to contains only alphabetical letters
        with single space between it.
    :param s:
        Single line of string, should be trimmed
    :return:
        A new line of string that is trimmed.
    """
    s = s.strip()
    characters = ascii_letters + " '"
    NonAlphabet = '''!()-[]{};:"\,<>./?@#$%^&*_~=0123456789+`|'''
    Astrophe = "'"
    Res = ""
    for char in s:
        if char in NonAlphabet:
            Res = Res + ' '
        elif char == Astrophe:
            continue  # Strip off apostrophe.
        else:
            Res += char if char in characters else ""
    return re.sub(' +', ' ', filter_upper(Res))
