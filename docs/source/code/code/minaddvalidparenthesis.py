def minAddToMakeValid(self, s: str) -> int:
	# keep a balance - every time the balance goes below 0
	# we could one and reset the balance to 0.
	extra_closing = 0
	extra_opening = 0
	for ch in s:
		if ch == '(':
			extra_opening += 1
		else:
			extra_opening -= 1
		if extra_opening < 0:
			extra_closing += 1 # counts extra closing paranthesis
			extra_opening = 0
	return extra_opening + extra_closing