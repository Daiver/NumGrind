# NumGrind
Simple computational graph with reverse mode autodiff framework

Not for production

#TODO

#Common
 - Add nice interface for GraphNodes
 - Switch from raw pointers to smart pointers

##Performance
 - add pre-allocated arrays for intermediate results
 - (long term issue) do not compute backwardPass for Constants

##Tests
 - split tests into several files
 - change tests names
