from string import ascii_letters

def trim_line(s:str, IgnoreCapitalzedWord=False):
    if IgnoreCapitalzedWord:
        s = ' '.join([word for word in s.split() if word[0].islower()])
    Res = ""
    for c in s:
        if c == "\'":
            continue
        Res += c if c in ascii_letters else (" " if len(Res) >= 1 and Res[-1] != ' ' else "")
    return Res.lower()


testline = "dkj 7& d  a7d7f7d7     ()03388kkkjhjk Alut ur Aoooirow T je Thhheus theusa. tHISissSOOOOO"
print(trim_line(testline))
print(trim_line(testline, True))


# def trim_line(s: str):
#     """
#         This function trims off all the punctuations in the line and collapse the
#         spaces in the text.
#         * The return string is going to contains only alphabetical letters
#         with single space between it.
#     :param s:
#         Single line of string, should be trimmed
#     :return:
#         A new line of string that is trimmed.
#     """
#     s = s.strip()
#     characters = ascii_letters + " '"
#     NonAlphabet = '''!()-[]{};:"\,<>./?@#$%^&*_~=0123456789+`|'''
#     Astrophe = "'"
#     Res = ""
#     for char in s:
#         if char in NonAlphabet:
#             Res = Res + ' '
#         elif char == Astrophe:
#             continue  # Strip off apostrophe.
#         else:
#             Res += char if char in characters else ""
#     return re.sub(' +', ' ', Res.lower())