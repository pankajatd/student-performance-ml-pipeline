import sys
from src.logger import logging
import traceback

def error_message_detail(error,error_detail :sys):
    _,_,exc_tb =error_detail.exc_info()
    file_name =exc_tb.tb_frame.f_code.co_filename
    error_message = (
        f"Error occurred in python script [{file_name}] "
        f"line number [{exc_tb.tb_lineno}] : {str(error)}"
    )
    
    return error_message
    
class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        # Store the full traceback
        detailed_message = error_message_detail(error_message, error_detail)
        logging.error(detailed_message)  # optional if using logging
        print("CustomException Traceback:")
        traceback.print_exc()  # this will print the full error trace in the terminal
        print("", detailed_message)
        super().__init__(detailed_message)
        self.error_message = detailed_message
            
        def __str__(self):
            return self.error_message


