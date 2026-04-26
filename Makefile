main:
	python analysis.py

onchain:
	python onchain_supplement.py

all: main onchain
