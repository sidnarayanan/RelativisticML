
from numpy import array

def writeArray1(ar,name):
	s=''
	counter=0
	for x in ar:
		s+='%s[%i]=%f\n'%(name,counter,x)
		counter+=1
        return s

def writeArray2(ar,name):
	s=''
	counterX=0
	counterY=0
	for x in ar:
		counterY=0
		for y in x:
			s+='%s[%i][%i]=%f\n'%(name,counterX,counterY,y)
			counterY+=1
		counterX+=1
        return s

class NetworkExporter(object):
  def __init__(self, classifier):
    self.classifier = classifier
    self.file = None
  def setFile(self,f):
    if type(f)==type(''):
      self.file = open(f,'w')
    else:
      self.file = f
  def export(self,name):
    if not self.file:
      print 'Please call NetworkExporter.setFile() first'
      return
    nIn = self.classifier.nIn
    nOut = self.classifier.nOut
    self.file.write("NeuralNet *%s = new NeuralNet(%i,%i);\n"%(name,nIn,nOut))
    parameters = self.classifier.getParameters()
    counter=0
    for p in parameters:
    	isLast = (counter==len(parameters)-1)
    	nIn = p['W'].shape[0]
    	nOut = p['W'].shape[1]
    	WName = '%s_W%i'%(name,counter)
    	bName = '%s_b%i'%(name,counter)
    	s = '''
float **%s = new float*[%i];
for (int i=0; i!=%s; ++i) {
	%s[i] = new float[%i];
}
float *%s = new float[%i];
'''%(WName,nIn,nIn,WName,nOut,bName,nOut)
    	s += writeArray1(p['b'],bName)
    	s += writeArray2(p['W'],WName)
    	s += '%s->AddLayer(%i,%i,%s,%s%s);\n'%(name,nIn,nOut,WName,bName,',true' if isLast else '')
    	self.file.write(s)
    	counter+=1
