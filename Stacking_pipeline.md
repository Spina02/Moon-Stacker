# Stacking pipeline


### pre
Cosmetic correction - CFA (filtri di colore)

Blink - deep sky stacker -> selezione le foto

### stack

batch preprocessing

	*bias*
		foto a tappo del sensore chiuso, iso = lights, t. esposizione minimo della camera, f indifferente

		*dark* - correggono hot pixels ed errori del sensore, tappo del sensore chiuso, t. esposizione = lights

	flat 
		rimuove rumore ed effetto vignettato
		- panno bianco sul sensore
		- luce equidistribuita
		- t. esposizione : istogramma -> al centro

	lights
		- foto effettive da mettere in stacking
		- deBayer (informarsi)

### proc
	


algoritmi
combination : average
rejection : winsorize Sigma Clipping