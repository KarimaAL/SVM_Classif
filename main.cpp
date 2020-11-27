#include "CData.h"
#include "CSVM.h"

int main(int, char**)
{
	CData m_data;
	CSVM m_svm;
    m_data.createData();
	m_svm.svmTraining(m_data.m_image);
	return 0;
}