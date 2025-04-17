using Plots
using LinearAlgebra

# Display the range Doppler map
fig = 124

# Create X and Y for the mesh grid
X = (1:size(RanDopMap, 2)) .* (2.998e8 / sr)
Y = -2.998e8 / refl["freq"] * newPRF / nPulses .* ((-nPulses / 2):(nPulses - 2) / 2)

# Set color limits
if value === nothing
    nPwr = -200
else
    nPwr = 20 * log10(minimum(abs.(value))) # nPwr goes to power of minimum detection
end

color_limits = (nPwr - 50, nPwr + 50)

# Plot the surface
p = plot(
    X, Y, 20 * log10.(abs.(RanDopMap)),
    st=:surface, c=:viridis,
    xlabel="Range", ylabel="Velocity",
    zlabel="Power (dB)", colorbar=true, zlims=color_limits,
	camera=(60, 20)
)

# Set axes limits
xlims!(0, maxRange)
ylims!(dopVec[end], dopVec[1])
zlims!(color_limits)

# Convert real-world coordinates to indices
range_indices = round.(Int, (adjustedExceedances[:, 2] .- minimum(X)) ./ step(X))
velocity_indices = round.(Int, (adjustedExceedances[:, 1] .- minimum(Y)) ./ step(Y))

# Ensure indices are within bounds
range_indices = clamp.(range_indices, 1, size(RanDopMap, 2))
velocity_indices = clamp.(velocity_indices, 1, size(RanDopMap, 1))

# Extract power at exceedance points
plotPower = [20 * log10(abs(RanDopMap[v, r])) for (v, r) in zip(velocity_indices, range_indices)]

# Highlight exceedances
scatter3d!(
    adjustedExceedances[:, 2], # Range values (original coordinates)
    adjustedExceedances[:, 1], # Velocity values (original coordinates)
    plotPower,
    marker=:circle, color=:red
)
