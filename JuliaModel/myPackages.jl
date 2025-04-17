
using Random
import Pkg
Pkg.add("FFTW")
Pkg.add("MAT")
Pkg.add("Plots")
Pkg.add("MeshGrid")
#Pkg.add("PyPlot")
Pkg.add("DataStructures")
Pkg.add("LinearAlgebra")
Pkg.add("DSP")
Pkg.add("Statistics")
Pkg.add("Interpolations")
Pkg.add("DataStructures")
using Interpolations
using Statistics
using LinearAlgebra
using FFTW
using Printf
using Base.Threads
using DSP
#using PyPlot
using Plots
using MeshGrid
include("combineExceedances.jl")
include("combineExceedances2.jl")
include("cfar.jl")
include("myFunctions.jl")
include("read_iq.jl")
