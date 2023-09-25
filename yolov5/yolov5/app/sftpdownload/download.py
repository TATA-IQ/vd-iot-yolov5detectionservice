
host = '172.16.0.171'
port = 22
username = 'aditya.singh'
password= 'Welcome#123'
import paramiko
from paramiko.transport import SecurityOptions, Transport
class SFTPClient():
 
    _connection = None
 
    def __init__(self, host, port, username, password):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.root_path="/models"
 
        self.OpenConnection(self.host, self.port,
                               self.username, self.password)
 
    @classmethod
    def OpenConnection(self, host, port, username, password):
 
        transport = Transport(sock=(host, port))
        transport.connect(username=username, password=password)
        self._connection = paramiko.SFTPClient.from_transport(transport)
    def download(self, remote_path, local_path):
        print("===>",remote_path)
        print("local path===>",local_path)
        self._connection.get(remote_path, local_path,
                             callback=None)
    def downloadpytorch(self,filenm,destpath):
        srcpath=self.root_path+"/pytorch/"+filenm
        destpath=destpath+filenm
        self.download(srcpath,destpath)
        
    def downloadtf(self,filenm,destpath):
        srcpath=self.root_path+"/tensorflow/"+filenm
        destpath=destpath
        self.download(srcpath,destpath)
        
    def downloadyolov5(self,filenm,destpath):
        srcpath=self.root_path+"/yolov5/"+filenm
        destpath=destpath
        self.download(srcpath,destpath)
        
    def downloadyolov8(self,srcpath,destpath):
        srcpath=self.root_path+"/yolov8/"+filenm
        destpath=destpath+filenm
        self.download(srcpath,destpath)
        
# sf=SFTPClient(host, port, username, password)
# sf.downloadtf("Person.zip","model/tg.zip")