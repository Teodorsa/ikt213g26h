from paddleocr import PaddleOCR

ocr = PaddleOCR(
    lang="en",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    enable_mkldnn=False,
)

image_path = "test_images/test.png"

results = ocr.predict(image_path)

for result in results:
    texts = result["rec_texts"]

    for text in texts:
        print(text)