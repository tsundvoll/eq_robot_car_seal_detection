import logging

import azure.functions as func


def main(myblob: func.InputStream, inputblob: bytes):
    logging.info(f"Python blob trigger function processed blob \n"
                 f"Name: {myblob.name}\n"
                 f"Blob Size: {myblob.length} bytes\n"
                 f"Blob Size: {len(inputblob)} bytes\n")

    logging.info(dir(inputblob))

    logging.info(inputblob)
