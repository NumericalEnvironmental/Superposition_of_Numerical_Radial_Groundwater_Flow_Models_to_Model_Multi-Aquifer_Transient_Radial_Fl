# Superposition_of_Numerical_Radial_Groundwater_Flow_Models_to_Model_Multi-Aquifer_Transient_Radial_Fl
This Python script provides the functionality of a multi-aquifer, transient radial groundwater flow model, similar to what would be expected from an equivalent analytical solution. This is accomplished through (1) numerical simulation, with pumping by layer, for radial flow in a multi-aquifer system with arbitrary layer thicknesses, conductivities, and storage coefficients, (2) interpolation of resulting drawdown as a function of radial distance and time in the form of an N X N matrix of interpolation functions (per monitoring layer, per pumping layer), (3) scaling relationships applied to pumping rates distributed across multiple layers, allocated by transmissivity and normalized by reference pumping rates used in the numerical modeling portion, and (4) superposition of solutions. A more detailed discussion of the conceptual model for this approach, along with an example application, are provided in my blog, [link pending].

The script requires the numpy, pandas, scipy (interpolate and sparse classes), and matplotlib packages. The following input files are also required:
* grid.txt – settings for model grid output and time series output at monitor points
* layers.csv – hydrologic properties of each layer, including elevation intervals
* monitors.csv – monitor points (x, y)
* params.txt – various model settings, including numerical radial flow solution constraints
* wells.csv – pumping well locations, screen intervals, pumping rates, and start/stop times

I'd appreciate hearing back from you if you find the code useful. Questions or comments are welcome at walt.mcnab@gmail.com.

THIS CODE/SOFTWARE IS PROVIDED IN SOURCE OR BINARY FORM "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

