###################################################################################
Distribution Shift
###################################################################################
Definitions
===================================================================================
.. note::
	* Distribution shift: :math:`p_{\text{train}}(\mathbf{x},y)\neq p_{\text{test}}(\mathbf{x},y)`
	* Covariate shift:

		* :math:`p_{\text{train}}(\mathbf{x})\neq p_{\text{test}}(\mathbf{x})`
		* :math:`p_{\text{train}}(y|\mathbf{x})=p_{\text{test}}(y|\mathbf{x})`
	* Concept shift:

		* :math:`p_{\text{train}}(\mathbf{x})=p_{\text{test}}(\mathbf{x})`
		* :math:`p_{\text{train}}(y|\mathbf{x})\neq p_{\text{test}}(y|\mathbf{x})`
	* Label shift:

		* Only in :math:`y\implies\mathbf{x}` problems.
		* :math:`p_{\text{train}}(y)\neq p_{\text{test}}(y)`
		* :math:`p_{\text{train}}(\mathbf{x}|y)=p_{\text{test}}(\mathbf{x}|y)`
