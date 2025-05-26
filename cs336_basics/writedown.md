
# Questions and answers

## unicode1
(a) What Unicode character does chr(0) return?
 What's returned is a null character \x00
(b) How does this character’s string representation (__repr__()) differ from its printed representation?
 \x00 vs emptiness. representation doesn't show anything since this character represent null.
(c) What happens when this character occurs in text? It may be helpful to play around with the
following in your Python interpreter and see if it matches your expectations:
>>> chr(0)
>>> print(chr(0))
>>> "this is a test" + chr(0) + "string"
>>> print("this is a test" + chr(0) + "string")

It's as if that character doesn't exist. It's not shown.

## unicode2

(a) What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather than
UTF-16 or UTF-32? It may be helpful to compare the output of these encodings for various
input strings.

utf-16 and utf-32 represent  a character in at least 2bytes and 4 bytes, respectively. Whereas, utf-8 will use 1 byte or more for each character. The main benefit of utf-8 is that the encoding leads to smaller amount of bytes in general as a character is represented into multiple only when it's required.

(b) Consider the following (incorrect) function, which is intended to decode a UTF-8 byte string into
a Unicode string. Why is this function incorrect? Provide an example of an input byte string
that yields incorrect results.
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])
>>> decode_utf8_bytes_to_str_wrong("hello".encode("utf-8"))
'hello'

Answer: It won't work because the method tries to decode a string byte by byte. This will not work for a character such as 😃 which requires 4 bytes to be represented.

Deliverable: An example input byte string for which decode_utf8_bytes_to_str_wrong produces incorrect output, with a one-sentence explanation of why the function is incorrect.
(c) Give a two byte sequence that does not decode to any Unicode character(s).

This wouldn't work: \xf5\xff, the reason is there are bytes that never appear in UTF-8 including 0xC0, 0xC1, 0xF5–0xFF

