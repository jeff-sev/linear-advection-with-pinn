# Compares a Fortran90 CFD solver against an equivalent Python/NumPy
# implementation, and trains a PINN (Python) as a neural-network surrogate
# for the same problem.
#
# Quick start:
#   make benchmark   # build everything, time both solvers, plot the result
#   make pinn        # train the neural network
#
# Override grid size, e.g.:  make run-fortran run-python NX=16000

PYTHON      := python3
VENV_DIR    := .venv
VENV_PYTHON := $(VENV_DIR)/bin/python
VENV_PIP    := $(VENV_DIR)/bin/pip

FC          := gfortran
FFLAGS      := -O3
FORTRAN_SRC := fortran/advection.f90
FORTRAN_BIN := bin/advection_fortran

PYTHON_CFD  := cfd/advection.py

NX          ?= 4000
SIZES       ?= 500 1000 2000 4000 8000
TRIALS      ?= 3

.PHONY: all venv fortran run-fortran run-python benchmark pinn clean distclean help

all: venv fortran

help:
	@echo "targets:"
	@echo "  make venv          create .venv and install requirements.txt"
	@echo "  make fortran       compile the Fortran90 CFD solver"
	@echo "  make run-fortran   run the Fortran solver once (NX=$(NX))"
	@echo "  make run-python    run the Python/NumPy solver once (NX=$(NX))"
	@echo "  make benchmark     time both solvers across SIZES and plot results"
	@echo "  make pinn          train the PINN neural-network surrogate"
	@echo "  make clean         remove build/run artifacts"
	@echo "  make distclean     clean + remove the virtualenv"

# --- Python virtual environment -------------------------------------------

venv: $(VENV_DIR)/.installed

$(VENV_DIR)/.installed: requirements.txt
	$(PYTHON) -m venv $(VENV_DIR)
	$(VENV_PIP) install --upgrade pip
	$(VENV_PIP) install -r requirements.txt
	touch $@

# --- Fortran90 CFD solver ---------------------------------------------------

fortran: $(FORTRAN_BIN)

$(FORTRAN_BIN): $(FORTRAN_SRC)
	mkdir -p bin
	$(FC) $(FFLAGS) -o $(FORTRAN_BIN) $(FORTRAN_SRC)

run-fortran: fortran | results
	./$(FORTRAN_BIN) $(NX)

# --- Python CFD solver -------------------------------------------------------

run-python: venv | results
	$(VENV_PYTHON) $(PYTHON_CFD) --nx $(NX)

# --- Benchmark: Fortran vs. Python runtime ----------------------------------

benchmark: venv fortran | results
	$(VENV_PYTHON) benchmark/compare.py \
		--fortran-bin $(FORTRAN_BIN) \
		--python-script $(PYTHON_CFD) \
		--sizes $(SIZES) \
		--trials $(TRIALS)

results:
	mkdir -p results

# --- Neural network (PINN) --------------------------------------------------

pinn: venv
	$(VENV_PYTHON) -m pinn.train

# --- Cleanup -----------------------------------------------------------------

clean:
	rm -rf bin results outputs
	rm -f fortran/*.mod
	find . -name "__pycache__" -type d -exec rm -rf {} +

distclean: clean
	rm -rf $(VENV_DIR)
