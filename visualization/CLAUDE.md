## Overview

The goal of this folder is to build a web application which can visualize LLMs. This includes understanding
the building blocks of the network, it's dimensions, and how they are connected. We'll start with this
more basic functionality and slowly build up. 

## Current State and Plan

At present, we only have an idea, no actual code or even concrete design of how this could work! Luckily, we are
are not starting from zero, as many previous libraries and implementations exist. 

For starters, we'll focus on just unpacking the necessary data from the model. We'll do this by loading in a model 
similar to how we did in the model test, and unpack the model architecture via text. 

Next, we'll want to save the necessary model architecture so we don't have to load the entire model into memory each time 
we want to simply visualize the architecture. To this end, we'll create a database which stores our different models and 
there information. 

After we have the models stored in a lightweight DB, we can progress with loading the model architecture and building visualization.
We'll start by creating a very basic block diagram to show the interconnectedness. 
