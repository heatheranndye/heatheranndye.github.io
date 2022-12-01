title: The Monty Hall Problem
author: Heather Ann Dye
date: 12/1/2022
category: simulation
tags: streamlit

## The Monty Hall Problem and Streamlit

Once I learned about [Streamlit.io](https://streamlit.io/), I couldn't resist the opportunity to write a
simulation of the Monty Hall problem!

The Monty Hall problem involves a game show and prizes! On the game show, contestants were given the choice of 
three doors. One door conceals a fabulous prize! 

Contests are asked to select a door. 

#### Then comes the twist -

The host opens one of the two remaining doors and asks if the contestant would like to switch to the remaining door. 

The question that emerges is: Switch or Stay? 

### What should the contestant do?

The contestent should go with the switch strategy - this will result in a win 66% of the time. In fact, the only way the
contestant looses with the switch strategy is if they originally selected the prize door. 

But why?  

##### With the Monty Hall problem, we can explore the answer in three different ways!

* Using Bayes Theorem and probability theory
* Actually hosting a game show and keeping track of the winners
* Constructing a simulation

Here's my [streamlit simulation](https://heatheranndye-montyhall-montymonty-app-t59nhv.streamlit.app/).

In this simulation, we construct a data frame to keep track of our game. 
For each trial, we need to perform four steps:

1. Selecting a door for the prize 
2. Selecting a door for the contestant's choice
3. Determine if the Stay strategy won (or not).
4. Determine if the Switch strategy won (or not).

In addition, we want to tabulate the percentage of wins go to each strategy over time. 

##### Steps 1 and 2:
We simulate the door choices using randomint.

```python
 select_door = random.randint(1,3)
```

#### Steps 3 and 4:

Write a short function to determine if the strategy won. Wins are
represented by 1's and losses by 0's. 

```python
def switch_strat(init_door: int, win_door: int)->int:
    """Switch strategy 
    Args:
        init_door (int): initial door selected
        win_door (int): prize door
    Returns:
        int: 0 or 1 to indicate win-loose
        """
    if init_door != win_door:
        return 1
    else: 
        return 0
```

#### Displaying the first 5 trials

The first five trials and there outcomes are displayed in a data frame. 
Simply looking at the first few trials can be deceptive, so 
the cumulative percentage wins are displayed in a graph.

We can add additional trials and see how the win percentages change as the number of trials 
increases. 