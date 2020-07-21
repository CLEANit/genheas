
import sys
import os.path
import numpy
import math
from Tools import Atoms
from pathlib import Path
#from Tools import PyVasp
#from Tools.Quaternions import Quaternion
import itertools
import  scipy
import h5py
from Tools import Constants
import mendeleev
import pybel
import molmod

class AbstractMolecule(object):

    def __init__(sefl):
        self.natoms = None
        self.atmasses = None
        self.atnums = None
        self.coordinates = None
        self.atsymbols = None
        self.atlabels = None




    def get_fragment(self, atomlist):
        frag = AbstractMolecule()

        frag.natoms = len(atomlist)
        frag.atmasses = self.atmasses[atomlist]
        frag.atnums = self.atnums[atomlist]
        frag.coordinates = self.coordinates[atomlist, :]

        frag.atsymbols = [self.atsymbols[i] for i in atomlist]
        return frag

    def __add__(self, mol2):
        twomol = AbstractMolecule()
        mol1 = self
        twomol.natoms = mol1.natoms + mol2.natoms
        twomol.atmasses = numpy.concatenate((mol1.atmasses, mol2.atmasses))
        twomol.atnums = numpy.concatenate((mol1.atnums, mol2.atnums))
        twomol.coordinates = numpy.concatenate((mol1.coordinates, mol2.coordinates))

        twomol.atsymbols = mol1.atsymbols + mol2.atsymbols
        return twomol

    def copy(self):
        mol = copy.deepcopy(self)
        return mol


    def get_coordinates_bohr(self):
        coords = self.coordinates
        return coords/Constants.Bohr_in_Angstrom

    def repulsion_energy(self):
        coords = self.get_coordinates_bohr()
        Z = self.atnums
        energies = []
        for i in range(self.natoms):
            for j in range(i+1, self.natoms):#j > i
                diffcoord = coords[i] -coords[j]
                energies.append(Z[i] *Z[j] /numpy.linalg.norm(diffcoord))
        energies = numpy.array(energies)
        return energies.sum()

    def get_centroid(self, weight=None, atomlist=None):
        """
        Param weight:list of weight of size natoms
        Param atomlist:list of atoms
        """
        if weight is None:
            weight = numpy.ones(self.natoms)
        if atomlist is None:
            atomlist = list(range(self.natoms))
        coords = self.coordinates
        if coords is None:
            return None
#        centroid = numpy.zeros((3, ))
#        for i in atomlist:
#            centroid[:]=centroid[:]+coords[i, :]*weight[i]
        centroid = numpy.dot(weight[atomlist], coords[atomlist])
        return centroid/weight[atomlist].sum()



    def get_center_of_mass(self, atomlist=None):
        mass = self.atmasses
        if mass is None:
            return None
        return self.get_centroid(weight=mass, atomlist=atomlist)


    def get_Rsquare_matrix(self, weight=None, atomlist=None):
        r"""
        Rsquare matrix is a matrix whose elements are given by:
        Rquare_ {\alpha\beta}= \sum_i  R_ {i\alpha} R_ {i\beta} weight_i
        where R_ {i\alpha} = coord[i, \alpha] - centroid
        Param weight:list of weight of size natoms
        Param atomlist:list of atoms
        """
        if weight is None:
            weight = numpy.ones(self.natoms)
        if atomlist is None:
            atomlist = list(range(self.natoms))
        coords = self.coordinates
        if coords is None:
            return None
        centroid = self.get_centroid(weight, atomlist)
        coords = coords[atomlist].copy() - centroid
        Rsquare = numpy.dot(coords.transpose() *weight[atomlist], coords)
        return Rsquare

    def get_moment_of_inertia(self, atomlist=None):
        r"""
        moi_ {\alpha\beta} = sum_i m_i[r^2_i \delta_ {\alpha\beta} - r_ {i\alpha} r_ {i\beta}]
         = (\sum_i m_i r^2_i)\delta_ {\alpha\beta} - Rsquare_ {\alpha \beta}
       moi = \sum_i m_i[\vec {r} _i \cdot  \vec {r} _i \identity - \vec {r} _i \otimes \vec {r} _i]
         """
        if atomlist is None:
            atomlist = list(range(self.natoms))
        mass = self.atmasses
        coords = self.coordinates
        if (coords is None) or (mass is None):
            return None
        R2 = self.get_Rsquare_matrix(weight=mass, atomlist=atomlist)
        return numpy.trace(R2) *numpy.identity(3) - R2


    def rotate_to_principle_axis(self, atomlist=None):
        moi = self.get_moment_of_inertia(atomlist)
        _, eigenvecs = numpy.linalg.eigh(moi)
        # ensure that we have a right-handed system of axes
        eigenvecs[:, 2] = numpy.cross(eigenvecs[:, 0], eigenvecs[:, 1])
        self.rotate(eigenvecs.transpose())



    def get_moment_of_charge(self, atomlist=None):
        """
        similar to moment of inertia but with the charge instead of the mass
        """
        if atomlist is None:
            atomlist = list(range(self.natoms))
        coords = self.coordinates
        charge = self.atnums
        if (coords is None) or (charge is None):
            return None
        R2 = self.get_Rsquare_matrix(weight=charge, atomlist=atomlist)
        return numpy.trace(R2) *numpy.identity(3) - R2


    def rotate_to_moment_of_charge(self, atomlist=None):
        moi = self.get_moment_of_charge(atomlist)
        _, eigenvecs = numpy.linalg.eigh(moi)
        # ensure that we have a right-handed system of axes
        eigenvecs[:, 2] = numpy.cross(eigenvecs[:, 0], eigenvecs[:, 1])
        self.rotate(eigenvecs.transpose())


    def translate(self, vec, atomlist=None):
        if atomlist is None:
            self.coordinates = self.coordinates + numpy.array(vec)
        else:
            self.coordinates[atomlist] = self.coordinates[atomlist] + numpy.array(vec)

    def rotate(self, rotmat):
        self.coordinates = numpy.dot(self.coordinates, rotmat.transpose())

    def scale(self, s):
        # s is a scaling factor
        self.coordinates = self.coordinates * s

    def write_to_xyz(self, filename='geom.xyz', title=None):
        coords = self.coordinates
        symbols = self.atsymbols
        atnums = self.atnums
        if title is None:
            title = filename
        if (coords is None) or ((symbols is None) and (atnums is None)):
            raise Exception("Cannot write xyz file: not enough informations")
        ofile = open(filename, 'w')
        ofile.write("{:d}\n".format(self.natoms))
        ofile.write("{}\n".format(title))
        if symbols is not None:
            for sym, coord in zip(symbols, coords):
                ofile.write("{}  {:12.6f} {:12.6f} {:12.6f}\n".format(sym, coord[0], coord[1], coord[2]))
        else:
            for an, coord in zip(atnums, coords):
                ofile.write("{:d}  {:12.6f} {:12.6f} {:12.6f}\n".format(an, coord[0], coord[1], coord[2]))
        ofile.write("\n")
        ofile.close()

    def isLinear(self):
        """
        Guess if a molecule is linear
        """
        coords = self.coordinates
        natoms = coords.shape[0]
        if natoms > 1:
            # vector along bond 1-2
            v12 = coords[0] -coords[1]
            v12 = v12/numpy.linalg.norm(v12)
            for coord in coords[2:]:
                v1i = coords[0] -coord
                v1i = v1i/numpy.linalg.norm(v1i)
                prod = numpy.dot(v12, v1i)# between -1 -> 1
                val = 1.0-abs(prod)# between 0 -> 1 where 0 mean linear
                if val > 1.0e-6:
                    return False
            return True
        else:
            return None # We have an atom

    def RMSD(self, mol2):
        """
        Calculate the Root Mean Square Deviation:
        Square root of the mean of the square of the distances between the atoms of two molecules
        """
        mol1 = self
        if mol1.natoms != mol2.natoms:
            print("Can't compare molecules with a different number of atoms")
            sys.exit(1)
        coord1 = mol1.coordinates
        coord2 = mol2.coordinates
        rmsd = ((coord1-coord2)**2).sum()
        rmsd = math.sqrt(rmsd / (mol1.natoms))
        return rmsd

    def distance(self, iatom, jatom):
        """Calculate the distance between two atoms of the same molecule
:param iatom, jatom:atom index(or list of atom indexes) zero-based
        return:one distance or a list of distances
        """
        if isinstance(iatom, int) or isinstance(iatom, numpy.int64):
            iatom = [iatom]
        if isinstance(jatom, int) or isinstance(jatom, numpy.int64):
            jatom = [jatom]
        if len(iatom) != len(jatom):
            raise Exception("distance: both lists should have the same length")
        coords = self.coordinates
        dist = []
        for i, j in zip(iatom, jatom):
            dist.append(numpy.linalg.norm(coords[i] -coords[j]))
        if len(dist) == 1:
            return dist[0]
        else:
            return dist

    @staticmethod
    def angle_vectors(vec1, vec2, deg=False):
        """Calculate the angle between two vectors.
:param vec1, vec2:two ndarray vectors
:return:the angle in radian
        """
        for vec in(vec1, vec2):
            if len(vec) != 3:
                raise Exception('angle_vectors:vectors are invalid')
        scalar_prod = numpy.dot(vec1, vec2)
        if abs(scalar_prod) < 1.0e-8:
            return numpy.pi

        val = scalar_prod / math.sqrt(numpy.dot(vec1, vec1) * numpy.dot(vec2, vec2))
        # sometimes acos behaves strange...
        if val > 1.:
            val = 1.
        elif val < -1.:
            val = -1
        angle = numpy.arccos(val)
        if deg:
            angle = numpy.degrees(angle)
        return angle

    def angle(self, iatom, jatom, katom, deg=False):
        """Calculate the angle between three atoms of the same molecule
:param iatom, jatom, katom:atom index(or list of atom indexes) zero-based
        return:one angle or a list of angles in radians
        """
        if isinstance(iatom, int) or isinstance(iatom, numpy.int64):
            iatom = [iatom]
        if isinstance(jatom, int) or isinstance(jatom, numpy.int64):
            jatom = [jatom]
        if isinstance(katom, int) or isinstance(katom, numpy.int64):
            katom = [katom]
        if (len(iatom) != len(jatom)) or (len(iatom) != len(katom)):
            raise Exception("distance: all lists should have the same length")
        coords = self.coordinates
        angle = []
        for i, j, k in zip(iatom, jatom, katom):
            angle.append(AbstractMolecule.angle_vectors(coords[i] -coords[j], coords[k] -coords[j], deg=deg))
        if len(angle) == 1:
            return angle[0]
        else:
            return angle


    def dihedral(self, iatom, jatom, katom, latom, deg=False):
        """Calculate the dihedral angle between four atoms of the same molecule.
:param iatom, jatom, katom, latom:atom index(or list of atom indexes) zero-based
        return:one torsion angle or a list of torsion angles in radians
        """
        if isinstance(iatom, int) or isinstance(iatom, numpy.int64):
            iatom = [iatom]
        if isinstance(jatom, int) or isinstance(jatom, numpy.int64):
            jatom = [jatom]
        if isinstance(katom, int) or isinstance(katom, numpy.int64):
            katom = [katom]
        if isinstance(latom, int) or isinstance(latom, numpy.int64):
            latom = [latom]
        if (len(iatom) != len(jatom)) or (len(iatom) != len(katom)) or (len(iatom) != len(latom)):
            raise Exception("distance: all lists should have the same length")
        coords = self.coordinates
        tangle = []
        for i, j, k, l in zip(iatom, jatom, katom, latom):
            v_ji = coords[i] - coords[j]
            v_jk = coords[k] - coords[j]
            v_kj = coords[j] - coords[k]
            v_kl = coords[l] - coords[k]
            # normal to the plane (i, j, k)
            norm1 = numpy.cross(v_ji, v_jk)
            # normal to the plane (j, k, l)
            norm2 = numpy.cross(v_kj, v_kl)
            # scalar triple product which defines the sign of the dihedral angle
            if numpy.dot(v_jk, numpy.cross(norm1, norm2)) < 0.0:
                sign = -1.
            else:
                sign = +1.
            tangle.append(sign * AbstractMolecule.angle_vectors(norm1, norm2, deg=deg))
        if len(tangle) == 1:
            return tangle[0]
        else:
            return tangle


    def distance_vect(self, i, j):
        """
        Return the distance vector between atom i and atom j
        """
        coords = self.coordinates

        return coords[j] -coords[i]

    def distances(self, mol2):
        """
        Return a list of vectors for the distances between each corresponding atoms
        """
        mol1 = self
        if mol1.natoms != mol2.natoms:
            print("Can't work on molecules with a different number of atoms")
            sys.exit(1)
        alist = []
        coord1 = mol1.coordinates
        coord2 = mol2.coordinates
        for iatom in range(mol1.natoms):
            alist.append(coord1[iatom] -coord2[iatom])
        return alist


    def distance_vect(self, i, j):
        """
        Return the distance vector between atom i and atom j
        """
        coords = self.coordinates

        return coords[j] -coords[i]

    def distances(self, mol2):
        """
        Return a list of vectors for the distances between each corresponding atoms
        """
        mol1 = self
        if mol1.natoms != mol2.natoms:
            print("Can't work on molecules with a different number of atoms")
            sys.exit(1)
        alist = []
        coord1 = mol1.coordinates
        coord2 = mol2.coordinates
        for iatom in range(mol1.natoms):
            alist.append(coord1[iatom] -coord2[iatom])
        return alist




    def nbatomtype(self, atnum):
        """
        Get the number of atoms of atomic number atnum in the molecule
        """
        atnums = self.atnums
        return numpy.where(atnums == atnum)[0].shape[0]

    def getatomtype(self, atnum):
        """
        Get the atoms of atomic number atnum in the molecule
        """
        atnums = self.atnums
        alist = numpy.where(atnums == atnum)[0]
        return self.get_fragment(alist)



    def get_bonds_in_molecule(self,coordi=None):


        if  coordi is None:
            coords = self.coordinates
        else:
            coords = coordi
        bonds = []
        dist = []
        natoms = coords.shape[0]

        if natoms > 1:
            for i in range(natoms):
                for j in range(i+1,natoms):
                    if i != j:
                        bonds.extend([[i,j]]) # a list of list
                        dist.append(numpy.linalg.norm(coords[i] -coords[j]))
            dist = numpy.array(dist)
            return bonds, dist
        else:
            return None , None# We have an atom









class MoleculeTools(AbstractMolecule):

    def __init__(self):
        self.masses = None
        self.numbers = None
        self.symbols = None

    def read_from_file(self, filename="file"):
        """
        Read the geometry from any file
        """
        # open the file if it exists
        if not os.path.isfile(filename):
            raise Exception("The file ['%s'] does not exist" % filename)
        ext = os.path.splitext(filename)[1].lower()

        if ext == ".data":
            self.read_from_data(filename=filename)
        elif  ext == ".log":
            f = open(filename, 'r')
            entete = f.readline()
            entete += f.readline()
            entete += f.readline()
            entete += f.readline()
            entete += f.readline()
            entete += f.readline()
            entete += f.readline()
            entete += f.readline()
            entete += f.readline()
            entete += f.readline()
            f.close()
            if entete.find("GAMESS") > -1:
                self.read_from_gamess(filename=filename)
            elif  entete.find('Gaussian') > -1:
                self.read_from_gaussian(filename=filename)
            else:
                raise Exception("The file [{}] is not a gamess or gaussian file".format(filename))
        elif  ext == ".out":
            f = open(filename, 'r')
            entete = f.readline()
            entete += f.readline()
            entete += f.readline()
            entete += f.readline()
            entete += f.readline()
            f.close()
            if entete.find("DALTON") > -1:
                self.read_from_dalton(filename=filename)
            else:
                self.read_from_molpro(filename=filename, oricoord=True)
        elif  ext == ".fchk":
            self.read_from_fchk(filename=filename)
        elif ext == ".xyz":
            self.read_from_xyz(filename=filename)
        elif ext == ".pdb":
            self.read_from_pdb(filename=filename)
        elif ext == ".cif":
            self.read_from_cif(filename=filename)
        elif ext == ".POSCAR":
            self.read_from_POSCAR(filename=filename)
        elif ext == ".hdf5":
            self.read_from_hdf5(filename=filename)
        else:
            try:
                self.read_from_nwchem(filename=filename)
            except:
                raise Exception("The file [{}] is not a xyz, cif, pdb, data file or a QM calculation file".format(filename))

        return

    def get_abstractMolecule(self, atomlist=None):
        if atomlist is None:
            atomlist = list(range(self.natoms))
        frag = AbstractMolecule()

        frag.natoms = len(atomlist)
        frag.atmasses = self.atmasses[atomlist]
        frag.atnums = self.atnums[atomlist]
        frag.coordinates = self.coordinates[atomlist, :]

        frag.atsymbols = [self.atsymbols[i] for i in atomlist]
        return frag

    def read_from_xyz(self, filename='geom.xyz'):
        try:
            mol=molmod.molecules.Molecule.from_file(filename)
            self.coordinates = mol.coordinates
            self.natoms = mol.size
            self.atsymbols = mol.symbols
        except:
            print('Error reading file')
