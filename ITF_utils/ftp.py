
# Import Module
import ftplib

# Ref Link: https://docs.python.org/3/library/ftplib.html

# Fill Required Information
HOSTNAME = "4.224.51.244"
USERNAME = "azureuser"
PASSWORD = "NumberTheory@54321"

# Connect FTP Server
ftp_server = ftplib.FTP(HOSTNAME, USERNAME, PASSWORD)

# force UTF-8 encoding
ftp_server.encoding = "utf-8"

ftp_server.mkd("TradeFinance/DocProcessing-26032026-process/")
ftp_server.mkd("TradeFinance/DocProcessing-26032026-process/Images")

# """
# with open('/home/tarun/NumberTheory/TradeFinance/Repos/final_delivery_sample_document/BundleApi/TradeFinance/WI_12345678/WI_123456$
#     ftp_server.retrbinary(' README', fp.write)
# """

with open("/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/aishiki_AlmasRaf_test_pdfs/pdf_files_test_set2/mohammedkm_23-01-2026_8-50-01.pdf", "rb") as local_file:
    ftp_server.storbinary(f'STOR /TradeFinance/DocProcessing-26032026-process/DocProcessing-26032026-process.pdf', local_file)

# with open('/home/tarun/NumberTheory/TradeFinance/Repos/final_delivery_sample_document/BundleApi/TradeFinance/WI_12345678/WI_12345678.pdf', 'wb') as fp:
#     ftp_server.retrbinary('RETR Images/Export-Bill_0.png', fp.write)
