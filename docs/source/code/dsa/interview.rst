######################################################################
Interview Day and Prior
######################################################################
.. attention::
	* [youtube.com] `Chris Jereza <https://www.youtube.com/watch?v=ksZ2wFRZ3gM>`_
	* [thatgirlcoder.com] `DSA Approach Format During Interviews <https://thatgirlcoder.com/>`_

**********************************************************************
Remember
**********************************************************************
#. Acting smart is VERY risky. Especially when you know how to solve the first version of the problem.
#. You DON'T need to remember how you solved it earlier. You CAN FIGURE IT OUT NOW.
#. NOT GIVING UP is what would help. Just follow the PROCESS.
#. If you mention alternate approaches, talk about the time/space complexity TRADE-OFF.
#. EVERY question you're asked is valid. It's also okay to NOT know something.
#. It's OKAY to screw up. Just follow the PROCESS.

**********************************************************************
The Process
**********************************************************************
#. Start with examples

	- Start with one small example. Specify input and output.
	- Confirm if the understanding of the problem is correct.
	- Confirm about the input range.
	- Identify corner cases.
	- Usually 3 examples are good - one for algorithm understanding, and 2 for corner cases.

#. Start noting down obversations and approach in comments

	- Lay out the algorithm in words.
	- Mention time complexity in the notes.

#. Code

	- Keep comments. Add code after it.

#. Test

	- Dry run on examples and debug.
#. Be prepared for follow ups.

**********************************************************************
Checklist
**********************************************************************
#. Evaluate trade-offs.  
#. Ask clarifying questions before coding - range, sorted, streaming.
#. Write the core logic.  
#. Check boundary conditions.  
#. Create test cases before dry running.  
#. Ensure a smooth implementation.  
#. Solve 3-4 follow-ups.  
#. Optimize time and space complexity.  
#. Use clear variable names and correct syntax.  
#. Wrap up efficiently.

**********************************************************************
During the interview: Remember
**********************************************************************
#. Take your time - longer than you think you should.

	Spend more time collaborating with the interviewer - planning, writing comments, discussing approach, brainstorming

#. Collect observations about the problem.

	Helps in understanding the nature of the problem. Identify base cases, edge cases.

#. Sample inputs and outputs

	Should be small enough - it should be possible to manually convert the input to output.

#. Write pseudocode

	#. Think out loud in comments and in pseudocode.
	#. During actual coding, swap comments with actual code.
	#. If the approach itself is not clear, it's okay to say: I am not sure about this. Need a min to think about it.

#. Approaching a solution
	
	#. Get a brute force to work - verifies that we're solving the right problem.
	
		If an optimization on this isn't immediately clear, ask if it's okay to code this approach.

	#. Think and talk about runtime and memory constantly
	
		Interviewer needs to know that you're thinking about efficiency. Might give points.

	#. Whenever stuck, think about this:
	
		#. Can we map it to something that we know how to solve?
		#. What exactly is required?
		#. What's the easiest way to get us there?
		#. What's the bottleneck? What do we need to keep track of?
		#. What other ways we can get there?

	#. Getting to an optimal solution
	
		#. One way to think about this - analyze current runtime. Check for data structures/approaches which are next better. Can any of them work?
		#. Think if stuck because the input isn't structured well for the task at hand.

			#. Can I preprocess the input so that the task gets done faster?
			#. We can either modify the algorithm to fit the data structure or modify the data structure to fit the algorithm. Latter is better.

#. Writing code

	#. Fill out the outline first - fill in details later
	#. Follow single responsibility principles (SRP)
	#. Keep in mind of variable naming and readability.

**********************************************************************
Prior to the interview: Remember
**********************************************************************
#. Take care of yourself

	#. Maintain motivation, confidence and mental health.
	#. Remember that you have other things going in life - getting this job would be good but not the end-goal of your life.
#. Leading up to the interview day?

	#. 7-10 days leading up to the interview

		#. Revisit all basic data structures and algorithms. Follow the process as you do.
		#. Try to form a picture in your head of how the approach works. This is what you'll remember.
		#. Pracice them 2-3 times if possible. Follow the process during this as well.
	#. 4-6 days leading up to the interview

		#. Work on problem solving approach for problems that you've already done.
		#. If feeling confident, try problems from the company tag. Else leave it.
	#. 2-3 days leading up to the interview

		#. Don't try to solve any new ridiculuously hard problem. Don't want to walk into the interview on a losing streak.
		#. Understand what makes you confident. Big picture, grasp on the bag of tricks, having the process reharsed and clear inside your head.
		#. It's okay if you cannot recall something. Read/watch the exact approach from book/video and understand space & time complexity.
	#. 24 hours leading up to the interview

		#. Clear up your mind from everything. Hit the gym. Listen to music and watch something. 
		#. Talk to people. Practice listening instead of being inside your head.
		#. Fast revisit of done problems (no code).
