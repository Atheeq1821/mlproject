import sys
from src.logger import logging
def handle_exception(error,error_details:sys):
    _,_,exc_tb=error_details.exc_info()
    file_name=exc_tb.tb_frame.f_code.co_filename
    error_msg="Error Occured in the file name [{0}] in line numer[{1}] and the error message is [{2}]".format(file_name,exc_tb.tb_lineno,str(error))
    return error_msg
class CustomeException(Exception):
    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_message)
        self.error_message=handle_exception(error_message,error_details=error_detail)
    def __str__(self) -> str:
        return self.error_message
    

'''if __name__=="__main__":
    try:
        a=2/0
    except Exception as e:
        logging.info("Divide by zero")
        raise CustomeEsception(e,sys)'''
