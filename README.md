so the idea behind checking if our car has crossed the finish line is a bit more tricky. For this check we just want to make sure that non of the intermediary squares that our car crosses over is a finishing square. We can do this by breaking down the car's steps into small increments, and then rounding every increment and then checking that square. This essentially means that we are tracing the line that the car takes, and we are seeing if the squares that approximate that line are the finishing squares. What this means is that we will have to conduct the out of bounds check after we conduct the finish line check, as there is a possibility that our car overshoots (but crosses) the finish line, and in that scenario we would want the car to finish, and not terminate.

actions will come in the form of a tuple, where the first integer will represent the change in the x velocity while the second integer will represent the change in the y velocity. For each of these the value of the integer can be a number from [-1,0,1]

the order of checks for step will be: 
1. unsure action is valid action
2. apply random change of no action
3. make sure updates to velocity is valid
4. update velocity
5. cap velocity
6. check to see if we crossed finish line (with current position and new velocity). If so give reward and end game if not continue
7. update position
8. check to see if current position is still within boundaries
9. if so then we move to the next step
10. if not then give reward and restart the game

due to the nature of how 2d arrays work in python, we will be going with y coord then x coord


the most intuitive way to implement the monte carlo agent's q table would be to just make a dictionary, where each key are the states (pos, vel, rew, end), and each of the values is essentially a dictionary of all possible moves, with predicted scores for each move.
This means that we can search up state action pairs like this: q[state][action] = value