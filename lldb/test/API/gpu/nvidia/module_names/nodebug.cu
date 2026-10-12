namespace physics {
namespace {
__global__ void integrate(int *data) { data[1] = data[0] + 1; }
} // namespace
} // namespace physics

void launchIntegrate(int *data) { physics::integrate<<<1, 1>>>(data); }
