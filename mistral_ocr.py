from mistralai import Mistral
import datauri
import os

api_key = os.environ["MISTRAL_API_KEY"]
client = Mistral(api_key=api_key)


def ocr_from_link(link):
    ocr_response = client.ocr.process(
        model="mistral-ocr-latest",
        document={
        "type": "document_url",
        "document_url": link,
        },
    )

def upload_pdf(filename):
  uploaded_pdf = client.files.upload(
    file={
      "file_name": filename,
      "content": open(filename, "rb"),
    },
    purpose="ocr"
  )
  signed_url = client.files.get_signed_url(file_id=uploaded_pdf.id)
  return signed_url.url

def create_markdown_file(ocr_response, output_filename = "output.md"):
  with open(output_filename, "wt") as f:
    for page in ocr_response.pages:
      f.write(page.markdown)
      for image in page.images:
        save_image(image)


if __name__ == "__main__":
    link = upload_pdf("data/SGD_19081909_0000214.pdf")
    print(f"link: {link}")
    ocr_response = ocr_from_link(link)
    print(ocr_response)
    create_markdown_file(ocr_response=ocr_response, output_filename= "data/19081909_historical_tweede_kammer.pdf")
