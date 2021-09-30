"""Handle input parsing and output writing."""

import tables
import anndata
import numpy as np
import scipy.sparse as sp
import scipy.io as io

from cellbender.remove_background import consts

from typing import Dict, Union, List, Optional
import logging
import os
import gzip


logger = logging.getLogger('')


def write_matrix_to_cellranger_h5(
        cellranger_version: int,
        output_file: str,
        gene_names: np.ndarray,
        barcodes: np.ndarray,
        inferred_count_matrix: sp.csc.csc_matrix,
        feature_types: Optional[np.ndarray] = None,
        gene_ids: Optional[np.ndarray] = None,
        genomes: Optional[np.ndarray] = None,
        cell_barcode_inds: Optional[np.ndarray] = None,
        ambient_expression: Optional[np.ndarray] = None,
        rho: Optional[np.ndarray] = None,
        z: Optional[np.ndarray] = None,
        d: Optional[np.ndarray] = None,
        p: Optional[np.ndarray] = None,
        phi: Optional[np.ndarray] = None,
        epsilon: Optional[np.ndarray] = None,
        fpr: Optional[float] = None,
        lambda_multiplier: Optional[float] = None,
        loss: Optional[Dict] = None) -> bool:
    """Write count matrix data to output HDF5 file using CellRanger format.

    Args:
        cellranger_version: Either 2 or 3. Determines the format of the output
            h5 file.
        output_file: Path to output .h5 file (e.g., 'output.h5').
        gene_names: Name of each gene (column of count matrix).
        gene_ids: Ensembl ID of each gene (column of count matrix).
        genomes: Name of the genome that each gene comes from.
        feature_types: Type of each feature (column of count matrix).
        barcodes: Name of each barcode (row of count matrix).
        inferred_count_matrix: Count matrix to be written to file, in sparse
            format.  Rows are barcodes, columns are genes.
        cell_barcode_inds: Indices into the original cell barcode array that
            were found to contain cells.
        ambient_expression: Vector of gene expression of the ambient RNA
            background counts that contaminate cell counts.
        rho: Hyperparameters for the contamination fraction distribution.
        epsilon: Latent encoding of droplet RT efficiency.
        z: Latent encoding of gene expression.
        d: Latent cell size scale factor.
        p: Latent probability that a barcode contains a cell.
        phi: Latent global overdispersion mean and scale.
        fpr: Target false positive rate for the regularized posterior denoised
            counts, where false positives are true counts that are (erroneously)
            removed.
        lambda_multiplier: The lambda multiplier value used to achieve the
            targeted false positive rate.
        loss: Training and test error, as ELBO, for each epoch.

    Note:
        To match the CellRanger .h5 files, the matrix is stored as its
        transpose, with rows as genes and cell barcodes as columns.

    """

    assert isinstance(inferred_count_matrix,
                      sp.csc_matrix), "The count matrix must be csc_matrix " \
                                      "format in order to write to HDF5."

    assert gene_names.size == inferred_count_matrix.shape[1], \
        "The number of gene names must match the number of columns in the " \
        "count matrix."

    if gene_ids is not None:
        assert gene_names.size == gene_ids.size, \
            f"The number of gene_names {gene_names.shape} must match " \
            f"the number of gene_ids {gene_ids.shape}."

    if feature_types is not None:
        assert gene_names.size == feature_types.size, \
            f"The number of gene_names {gene_names.shape} must match " \
            f"the number of feature_types {feature_types.shape}."

    if genomes is not None:
        assert gene_names.size == genomes.size, \
            "The number of gene_names must match the number of genome designations."

    assert barcodes.size == inferred_count_matrix.shape[0], \
        "The number of barcodes must match the number of rows in the count" \
        "matrix."

    # This reverses the role of rows and columns, to match CellRanger format.
    inferred_count_matrix = inferred_count_matrix.transpose().tocsc()

    # Write to output file.
    try:
        with tables.open_file(output_file, "w",
                              title="CellBender remove-background output") as f:

            if cellranger_version == 2:

                # Create the group where data will be stored.
                group = f.create_group("/", "background_removed",
                                       "Counts after background correction")

                # Create arrays within that group for gene info.
                f.create_array(group, "gene_names", gene_names)
                if gene_ids is not None:
                    f.create_array(group, "genes", gene_ids)

            elif cellranger_version == 3:

                # Create the group where data will be stored: name is "matrix".
                group = f.create_group("/", "matrix",
                                       "Counts after background correction")

                # Create a sub-group called "features"
                feature_group = f.create_group(group, "features",
                                               "Genes and other features measured")

                # Create arrays within that group for feature info.
                f.create_array(feature_group, "name", gene_names)
                if gene_ids is not None:
                    f.create_array(feature_group, "id", gene_ids)
                if feature_types is not None:
                    f.create_array(feature_group, "feature_type", feature_types)
                if genomes is not None:
                    f.create_array(feature_group, "genome", genomes)

                # Copy the other extraneous information from the input file.
                # (Some user might need it for some reason.)
                # TODO

            else:
                raise NotImplementedError(f'Trying to save to CellRanger '
                                          f'v{cellranger_version} format, which '
                                          f'is not implemented.')

            # Code for both versions.
            f.create_array(group, "barcodes", barcodes)

            # Create arrays to store the count data.
            f.create_array(group, "data", inferred_count_matrix.data)
            f.create_array(group, "indices", inferred_count_matrix.indices)
            f.create_array(group, "indptr", inferred_count_matrix.indptr)
            f.create_array(group, "shape", inferred_count_matrix.shape)

            # Store background gene expression, barcode_inds, z, d, and p.
            if cell_barcode_inds is not None:
                f.create_array(group, "barcode_indices_for_latents",
                               cell_barcode_inds)
            if ambient_expression is not None:
                f.create_array(group, "ambient_expression", ambient_expression)
            if z is not None:
                f.create_array(group, "latent_gene_encoding", z)
            if d is not None:
                f.create_array(group, "latent_scale", d)
            if p is not None:
                f.create_array(group, "latent_cell_probability", p)
            if phi is not None:
                f.create_array(group, "overdispersion_mean_and_scale", phi)
            if rho is not None:
                f.create_array(group, "contamination_fraction_params", rho)
            if epsilon is not None:
                f.create_array(group, "latent_RT_efficiency", epsilon)
            if fpr is not None:
                f.create_array(group, "target_false_positive_rate", fpr)
            if lambda_multiplier is not None:
                f.create_array(group, "lambda_multiplier", lambda_multiplier)
            if loss is not None:
                f.create_array(group, "training_elbo_per_epoch",
                               np.array(loss['train']['elbo']))
                if 'test' in loss.keys():
                    f.create_array(group, "test_elbo",
                                   np.array(loss['test']['elbo']))
                    f.create_array(group, "test_epoch",
                                   np.array(loss['test']['epoch']))
                    f.create_array(group, "fraction_data_used_for_testing",
                                   1. - consts.TRAINING_FRACTION)

        logging.info(f"Succeeded in writing CellRanger v{cellranger_version} "
                     f"format output to file {output_file}")

        return True

    except Exception:
        logging.warning(f"Encountered an error writing output to file "
                        f"{output_file}.  "
                        f"Output may be incomplete.")

        return False


def load_data(input_file: str)\
        -> Dict[str, Union[sp.csr.csr_matrix, List[np.ndarray], np.ndarray]]:
    """Load a dataset into the SingleCellRNACountsDataset object from
    the self.input_file"""

    # Detect input data type.
    data_type = detect_input_data_type(input_file=input_file)

    # Load the dataset.
    if data_type == 'cellranger_mtx':

        logger.info(f"Loading data from directory {input_file}")
        data = get_matrix_from_cellranger_mtx(input_file)

    elif data_type == 'cellranger_h5':

        logger.info(f"Loading data from file {input_file}")
        data = get_matrix_from_cellranger_h5(input_file)

    elif data_type == 'dropseq_dge':

        logger.info(f"Loading data from file {input_file}")
        data = get_matrix_from_dropseq_dge(input_file)

    elif data_type == 'bd_rhapsody':

        logger.info(f"Loading data from file {input_file}")
        data = get_matrix_from_bd_rhapsody(input_file)

    elif data_type == 'anndata':

        logger.info(f"Loading data from file {input_file}")
        data = get_matrix_from_anndata(input_file)

    else:
        raise ValueError(f'Input data type {data_type} is not recognized.')

    return data


def detect_input_data_type(input_file: str) -> str:
    """Detect the type of input data."""

    # Error if no input data file has been specified.
    assert input_file is not None, \
        "Attempting to load data, but no input file was specified."

    file_ext = os.path.splitext(input_file)[1]

    # Detect type.
    if os.path.isdir(input_file):
        return 'cellranger_mtx'

    elif file_ext == '.h5':
        return 'cellranger_h5'

    elif input_file.endswith('.txt.gz') or input_file.endswith('.txt'):
        return 'dropseq_dge'

    elif input_file.endswith('.csv.gz') or input_file.endswith('.csv'):
        return 'bd_rhapsody'

    elif input_file.endswith('.h5ad'):
        return 'anndata'

    else:
        raise ValueError('Failed to determine input file type for '
                         + input_file + '\n'
                         + 'This must either be: a directory that contains '
                           'CellRanger-format MTX outputs; a single CellRanger '
                           '".h5" file; a DropSeq-format DGE ".txt.gz" file; '
                           'or a BD-Rhapsody-format ".csv" file')


def detect_cellranger_version_mtx(filedir: str) -> int:
    """Detect which version of CellRanger (2 or 3) created this mtx directory.

    Args:
        filedir: string path to .mtx file that contains the raw gene
            barcode matrix in a sparse coo text format.

    Returns:
        CellRanger version, either 2 or 3, as an integer.

    """

    assert os.path.isdir(filedir), f"The directory {filedir} is not accessible."

    if os.path.isfile(os.path.join(filedir, 'features.tsv.gz')):
        return 3

    else:
        return 2


def detect_cellranger_version_h5(filename: str) -> int:
    """Detect which version of CellRanger (2 or 3) created this h5 file.

    Args:
        filename: string path to .mtx file that contains the raw gene
            barcode matrix in a sparse coo text format.

    Returns:
        version: CellRanger version, either 2 or 3, as an integer.

    """

    with tables.open_file(filename, 'r') as f:

        # For CellRanger v2, each group in the table (other than root)
        # contains a genome.
        # For CellRanger v3, there is a 'matrix' group that contains 'features'.

        version = 2

        try:

            # This works for version 3 but not for version 2.
            getattr(f.root.matrix, 'features')
            version = 3

        except tables.NoSuchNodeError:
            pass

    return version


def get_matrix_from_cellranger_mtx(filedir: str) \
        -> Dict[str, Union[sp.csr.csr_matrix, List[np.ndarray], np.ndarray]]:
    """Load a count matrix from an mtx directory from CellRanger's output.

    For CellRanger v2:
    The directory must contain three files:
        matrix.mtx
        barcodes.tsv
        genes.tsv

    For CellRanger v3:
    The directory must contain three files:
        matrix.mtx.gz
        barcodes.tsv.gz
        features.tsv.gz

    This function returns a dictionary that includes the count matrix, the gene
    names (which correspond to columns of the count matrix), and the barcodes
    (which correspond to rows of the count matrix).

    Args:
        filedir: string path to .mtx file that contains the raw gene
            barcode matrix in a sparse coo text format.

    Returns:
        out['matrix']: scipy.sparse.csr.csr_matrix of unique UMI counts, with
            barcodes as rows and genes as columns
        out['barcodes']: numpy array of strings which are the nucleotide
            sequences of the barcodes that correspond to the rows in
            the out['matrix']
        out['gene_names']: List of numpy arrays, where the number of elements
            in the list is the number of genomes in the dataset.  Each numpy
            array contains the string names of genes in the genome, which
            correspond to the columns in the out['matrix'].
        out['gene_ids']: List of numpy arrays, where the number of elements
             in the list is the number of genomes in the dataset.  Each numpy
             array contains the string Ensembl ID of genes in the genome, which
             also correspond to the columns in the out['matrix'].
        out['feature_types']: List of numpy arrays, where the number of elements
             in the list is the number of genomes in the dataset.  Each numpy
             array contains the string feature types of genes (or possibly
             antibody capture reads), which also correspond to the columns
             in the out['matrix'].

    """

    assert os.path.isdir(filedir), "The directory {filedir} is not accessible."

    # Decide whether data is CellRanger v2 or v3.
    cellranger_version = detect_cellranger_version_mtx(filedir=filedir)
    logger.info(f"CellRanger v{cellranger_version} format")

    # CellRanger version 3
    if cellranger_version == 3:

        matrix_file = os.path.join(filedir, 'matrix.mtx.gz')
        gene_file = os.path.join(filedir, 'features.tsv.gz')
        barcode_file = os.path.join(filedir, 'barcodes.tsv.gz')

        # Read in feature names.
        features = np.genfromtxt(fname=gene_file,
                                 delimiter="\t", skip_header=0,
                                 dtype='<U100')

        # Read in gene expression and feature data.
        gene_ids = features[:, 0].squeeze()  # first column
        gene_names = features[:, 1].squeeze()  # second column
        feature_types = features[:, 2].squeeze()  # third column

    # CellRanger version 2
    elif cellranger_version == 2:

        # Read in the count matrix using scipy.
        matrix_file = os.path.join(filedir, 'matrix.mtx')
        gene_file = os.path.join(filedir, "genes.tsv")
        barcode_file = os.path.join(filedir, "barcodes.tsv")

        # Read in gene names.
        gene_data = np.genfromtxt(fname=gene_file,
                                  delimiter="\t", skip_header=0,
                                  dtype='<U100')
        if len(gene_data.shape) == 1:  # custom file format with just gene names
            gene_names = gene_data.squeeze()
            gene_ids = None
        else:  # the 10x CellRanger v2 format with two columns
            gene_names = gene_data[:, 1].squeeze()  # second column
            gene_ids = gene_data[:, 0].squeeze()  # first column
        feature_types = None

    else:
        raise NotImplementedError('MTX format was not identifiable as CellRanger '
                                  'v2 or v3.  Please check 10x Genomics formatting.')

    # For both versions:

    # Read in sparse count matrix.
    count_matrix = io.mmread(matrix_file).tocsr().transpose()

    # Read in barcode names.
    barcodes = np.genfromtxt(fname=barcode_file,
                             delimiter="\t", skip_header=0, dtype='<U20')

    # Issue warnings if necessary, based on dimensions matching.
    if count_matrix.shape[1] != len(gene_names):
        logger.warning(f"Number of gene names in {filedir}/genes.tsv "
                        f"does not match the number expected from the "
                        f"count matrix.")
    if count_matrix.shape[0] != len(barcodes):
        logger.warning(f"Number of barcodes in {filedir}/barcodes.tsv "
                        f"does not match the number expected from the "
                        f"count matrix.")

    return {'matrix': count_matrix,
            'gene_names': gene_names,
            'feature_types': feature_types,
            'gene_ids': gene_ids,
            'genomes': None,  # TODO: check if this info is available in either version
            'barcodes': barcodes,
            'cellranger_version': cellranger_version}


def get_matrix_from_cellranger_h5(filename: str) \
        -> Dict[str, Union[sp.csr.csr_matrix, np.ndarray]]:
    """Load a count matrix from an h5 file from CellRanger's output.

    The file needs to be a _raw_gene_bc_matrices_h5.h5 file.  This function
    returns a dictionary that includes the count matrix, the gene names (which
    correspond to columns of the count matrix), and the barcodes (which
    correspond to rows of the count matrix).

    This function works for CellRanger v2 and v3 HDF5 formats.

    Args:
        filename: string path to .h5 file that contains the raw gene
            barcode matrices

    Returns:
        out['matrix']: scipy.sparse.csr.csr_matrix of unique UMI counts, with
            barcodes as rows and genes as columns
        out['barcodes']: numpy array of strings which are the nucleotide
            sequences of the barcodes that correspond to the rows in
            the out['matrix']
        out['gene_names']: List of numpy arrays, where the number of elements
            in the list is the number of genomes in the dataset.  Each numpy
            array contains the string names of genes in the genome, which
            correspond to the columns in the out['matrix'].
        out['gene_ids']: List of numpy arrays, where the number of elements
             in the list is the number of genomes in the dataset.  Each numpy
             array contains the string Ensembl ID of genes in the genome, which
             also correspond to the columns in the out['matrix'].
        out['feature_types']: List of numpy arrays, where the number of elements
             in the list is the number of genomes in the dataset.  Each numpy
             array contains the string feature types of genes (or possibly
             antibody capture reads), which also correspond to the columns
             in the out['matrix'].

    """

    # Detect CellRanger version.
    cellranger_version = detect_cellranger_version_h5(filename=filename)
    logger.info(f"CellRanger v{cellranger_version} format")

    with tables.open_file(filename, 'r') as f:
        # Initialize empty lists.
        csc_list = []
        barcodes = None
        feature_ids = None
        feature_types = None
        genomes = None

        # CellRanger v2:
        # Each group in the table (other than root) contains a genome,
        # so walk through the groups to get data for each genome.
        if cellranger_version == 2:

            feature_names = []
            feature_ids = []
            genomes = []

            for group in f.walk_groups():
                try:
                    # Read in data for this genome, and put it into a
                    # scipy.sparse.csc.csc_matrix
                    barcodes = getattr(group, 'barcodes').read()
                    data = getattr(group, 'data').read()
                    indices = getattr(group, 'indices').read()
                    indptr = getattr(group, 'indptr').read()
                    shape = getattr(group, 'shape').read()
                    csc_list.append(sp.csc_matrix((data, indices, indptr),
                                                  shape=shape))
                    fnames_this_genome = getattr(group, 'gene_names').read()
                    feature_names.extend(fnames_this_genome)
                    feature_ids.extend(getattr(group, 'genes').read())
                    genomes.extend([group._g_gettitle()] * fnames_this_genome.size)

                except tables.NoSuchNodeError:
                    # This exists to bypass the root node, which has no data.
                    pass

            # Create numpy arrays.
            feature_names = np.array(feature_names, dtype=str)
            genomes = np.array(genomes, dtype=str)
            if len(feature_ids) > 0:
                feature_ids = np.array(feature_ids)
            else:
                feature_ids = None

        # CellRanger v3:
        # There is only the 'matrix' group.
        elif cellranger_version == 3:

            # Read in data for this genome, and put it into a
            # scipy.sparse.csc.csc_matrix
            barcodes = getattr(f.root.matrix, 'barcodes').read()
            data = getattr(f.root.matrix, 'data').read()
            indices = getattr(f.root.matrix, 'indices').read()
            indptr = getattr(f.root.matrix, 'indptr').read()
            shape = getattr(f.root.matrix, 'shape').read()
            csc_list.append(sp.csc_matrix((data, indices, indptr),
                                          shape=shape))

            # Read in 'feature' information
            feature_group = f.get_node(f.root.matrix, 'features')
            feature_names = getattr(feature_group, 'name').read()

            try:
                feature_types = getattr(feature_group, 'feature_type').read()
            except tables.NoSuchNodeError:
                # This exists in case someone produced a file without feature_type.
                pass
            try:
                feature_ids = getattr(feature_group, 'id').read()
            except tables.NoSuchNodeError:
                # This exists in case someone produced a file without feature id.
                pass
            try:
                genomes = getattr(feature_group, 'genome').read()
            except tables.NoSuchNodeError:
                # This exists in case someone produced a file without feature genome.
                pass

    # Put the data together (possibly from several genomes for v2 datasets).
    count_matrix = sp.vstack(csc_list, format='csc')
    count_matrix = count_matrix.transpose().tocsr()

    # Issue warnings if necessary, based on dimensions matching.
    if count_matrix.shape[1] != feature_names.size:
        logger.warning(f"Number of gene names ({feature_names.size}) in {filename} "
                       f"does not match the number expected from the count "
                       f"matrix ({count_matrix.shape[1]}).")
    if count_matrix.shape[0] != barcodes.size:
        logger.warning(f"Number of barcodes ({barcodes.size}) in {filename} "
                       f"does not match the number expected from the count "
                       f"matrix ({count_matrix.shape[0]}).")

    return {'matrix': count_matrix,
            'gene_names': feature_names,
            'gene_ids': feature_ids,
            'genomes': genomes,
            'feature_types': feature_types,
            'barcodes': barcodes,
            'cellranger_version': cellranger_version}


def get_matrix_from_dropseq_dge(filename: str) \
        -> Dict[str, Union[sp.csr.csr_matrix, np.ndarray]]:
    """Load a count matrix from a DropSeq DGE matrix file.

    The file needs to be a gzipped text file in DGE format.  This function
    returns a dictionary that includes the count matrix, the gene names (which
    correspond to columns of the count matrix), and the barcodes (which
    correspond to rows of the count matrix).  Reads in the file line by line
    instead of trying to read in an entire dense matrix at once, which might
    require quite a bit of memory.

    Args:
        filename: string path to .txt.gz file that contains the raw gene
            barcode matrix

    Returns:
        out['matrix']: scipy.sparse.csr.csr_matrix of unique UMI counts, with
            barcodes as rows and genes as columns
        out['barcodes']: numpy array of strings which are the nucleotide
            sequences of the barcodes that correspond to the rows in
            the out['matrix']
        out['gene_names']: List of numpy arrays, where the number of elements
            in the list is the number of genomes in the dataset.  Each numpy
            array contains the string names of genes in the genome, which
            correspond to the columns in the out['matrix'].
        out['gene_ids']: List of numpy arrays, where the number of elements
            in the list is the number of genomes in the dataset.  Each numpy
            array contains the string Ensembl ID of genes in the genome, which
            also correspond to the columns in the out['matrix'].

    """

    logger.info(f"DropSeq DGE format")

    load_fcn = gzip.open if filename.endswith('.gz') else open

    with load_fcn(filename, 'rt') as f:

        # Skip the comment '#' lines in header
        for header in f:
            if header[0] == '#':
                continue
            else:
                break

        # Read in first row with droplet barcodes
        barcodes = header.split('\n')[0].split('\t')[1:]

        # Gene names are first entry per row
        gene_names = []

        # Arrays used to construct a sparse matrix
        row = []
        col = []
        data = []

        # Read in rest of file row by row
        for i, line in enumerate(f):
            # Parse row into gene name and count data
            parsed_line = line.split('\n')[0].split('\t')
            gene_names.append(parsed_line[0])
            counts = np.array(parsed_line[1:], dtype=int)

            # Create sparse version of data and add to arrays
            nonzero_col_inds = np.nonzero(counts)[0]
            row.extend([i] * nonzero_col_inds.size)
            col.extend(nonzero_col_inds)
            data.extend(counts[nonzero_col_inds])

    count_matrix = sp.csc_matrix((data, (row, col)),
                                 shape=(len(gene_names), len(barcodes)),
                                 dtype=float).transpose()

    return {'matrix': count_matrix,
            'gene_names': np.array(gene_names),
            'gene_ids': None,
            'genomes': None,
            'feature_types': None,
            'barcodes': np.array(barcodes)}


def get_matrix_from_bd_rhapsody(filename: str) \
        -> Dict[str, Union[sp.csr.csr_matrix, np.ndarray]]:
    """Load a count matrix from a BD Rhapsody MolsPerCell.csv file.

    The file needs to be in MolsPerCell_Unfiltered format, which is comma
    separated, where rows are barcodes and columns are genes.  Can be gzipped
    or not.  This function returns a dictionary that includes the count matrix,
    the gene names (which correspond to columns of the count matrix), and the
    barcodes (which correspond to rows of the count matrix).  Reads in the file
    line by line instead of trying to read in an entire dense matrix at once,
    which might require quite a bit of memory.

    Args:
        filename: string path to .csv file that contains the raw gene
            barcode matrix MolsPerCell_Unfiltered.csv

    Returns:
        out['matrix']: scipy.sparse.csr.csr_matrix of unique UMI counts, with
            barcodes as rows and genes as columns
        out['barcodes']: numpy array of strings which are the nucleotide
            sequences of the barcodes that correspond to the rows in
            the out['matrix']
        out['gene_names']: List of numpy arrays, where the number of elements
            in the list is the number of genomes in the dataset.  Each numpy
            array contains the string names of genes in the genome, which
            correspond to the columns in the out['matrix'].
        out['gene_ids']: List of numpy arrays, where the number of elements
            in the list is the number of genomes in the dataset.  Each numpy
            array contains the string Ensembl ID of genes in the genome, which
            also correspond to the columns in the out['matrix'].

    """

    logger.info(f"BD Rhapsody MolsPerCell_Unfiltered.csv format")

    load_fcn = gzip.open if filename.endswith('.gz') else open

    with load_fcn(filename, 'rt') as f:

        # Skip the comment '#' lines in header
        for header in f:
            if header[0] == '#':
                continue
            else:
                break

        # Read in first row with gene names
        gene_names = header.split('\n')[0].split(',')[1:]

        # Barcode names are first entry per row
        barcodes = []

        # Arrays used to construct a sparse matrix
        row = []
        col = []
        data = []

        # Read in rest of file row by row
        for i, line in enumerate(f):
            # Parse row into gene name and count data
            parsed_line = line.split('\n')[0].split(',')
            barcodes.append(parsed_line[0])
            counts = np.array(parsed_line[1:], dtype=np.int)

            # Create sparse version of data and add to arrays
            nonzero_col_inds = np.nonzero(counts)[0]
            row.extend([i] * nonzero_col_inds.size)
            col.extend(nonzero_col_inds)
            data.extend(counts[nonzero_col_inds])

    count_matrix = sp.csc_matrix((data, (row, col)),
                                 shape=(len(barcodes), len(gene_names)),
                                 dtype=np.float)

    return {'matrix': count_matrix,
            'gene_names': np.array(gene_names),
            'gene_ids': None,
            'genomes': None,
            'feature_types': None,
            'barcodes': np.array(barcodes)}


def get_matrix_from_anndata(filename: str) \
        -> Dict[str, Union[sp.csr.csr_matrix, np.ndarray]]:
    """Load a count matrix from an h5ad AnnData file.
    The file needs to contain raw counts for all measured barcodes in the
    `.X` attribute or a `.layer[{'counts', 'spliced'}]` attribute.  This function
    returns a dictionary that includes the count matrix, the gene names (which
    correspond to columns of the count matrix), and the barcodes (which
    correspond to rows of the count matrix).
    This function works for any AnnData object meeting the above requirements,
    as generated by alignment methods like `kallisto | bustools`.
    Args:
        filename: string path to .h5ad file that contains the raw gene
            barcode matrices
    Returns:
        out['matrix']: scipy.sparse.csr.csr_matrix of unique UMI counts, with
            barcodes as rows and genes as columns
        out['barcodes']: numpy array of strings which are the nucleotide
            sequences of the barcodes that correspond to the rows in
            the out['matrix']
        out['gene_names']: List of numpy arrays, where the number of elements
            in the list is the number of genomes in the dataset.  Each numpy
            array contains the string names of genes in the genome, which
            correspond to the columns in the out['matrix'].
        out['gene_ids']: List of numpy arrays, where the number of elements
             in the list is the number of genomes in the dataset.  Each numpy
             array contains the string Ensembl ID of genes in the genome, which
             also correspond to the columns in the out['matrix'].
        out['feature_types']: List of numpy arrays, where the number of elements
             in the list is the number of genomes in the dataset.  Each numpy
             array contains the string feature types of genes (or possibly
             antibody capture reads), which also correspond to the columns
             in the out['matrix'].
    """
    logger.info(f"Detected AnnData format")

    adata = anndata.read_h5ad(filename)

    if "counts" in adata.layers.keys():
        # this is a common manual setting for users of scVI
        # given the manual convention, we prefer this matrix to
        # .X since it is less likely to represent something other
        # than counts
        logger.info("Found `.layers['counts']`. Using for count data.")
        count_matrix = adata.layers["counts"]
    elif "spliced" in adata.layers.keys() and adata.X is None:
        # alignment using kallisto | bustools with intronic counts
        # does not populate `.X` by default, but does populate
        # `.layers['spliced'], .layers['unspliced']`.
        # we use spliced counts for analysis
        logger.info("Found `.layers['spliced']`. Using for count data.")
        count_matrix = adata.layers["spliced"]
    else:
        logger.info("Using `.X` for count data.")
        count_matrix = adata.X

    # check that `count_matrix` contains a large number of barcodes,
    # consistent with a raw single cell experiment
    if count_matrix.shape[0] < consts.MINIMUM_BARCODES_H5AD:
        # this experiment might be prefiltered
        msg = f"Only {count_matrix.shape[0]} barcodes were found.\n"
        msg += "This suggests the matrix was prefiltered.\n"
        msg += "CellBender requires a raw, unfiltered [Barcodes, Genes] matrix."
        logger.warning(msg)

    # AnnData is [Cells, Genes], no need to transpose
    # we typecast explicitly in the off chance `count_matrix` was dense.
    count_matrix = sp.csr_matrix(count_matrix)
    # feature names and ids are not consistently delineated in AnnData objects
    # so we attempt to find relevant features using common values.
    feature_names = np.array(adata.var_names, dtype=str)
    barcodes = np.array(adata.obs_names, dtype=str)

    # Make an attempt to find feature_IDs if they are present.
    feature_ids = None
    for key in ['gene_id', 'gene_ids']:
        if key in adata.var.keys():
            feature_ids = np.array(adata.var[key].values, dtype=str)

    # Make an attempt to find feature_types if they are present.
    feature_types = None
    for key in ['feature_type', 'feature_types']:
        if key in adata.var.keys():
            feature_types = np.array(adata.var[key].values, dtype=str)

    # Make an attempt to find genomes if they are present.
    genomes = None
    for key in ['genome', 'genomes']:
        if key in adata.var.keys():
            genomes = np.array(adata.var['genomes'].values, dtype=str)

    # Issue warnings if necessary, based on dimensions matching.
    if count_matrix.shape[1] != feature_names.size:
        logger.warning(f"Number of gene names ({feature_names.size}) in {filename} "
                       f"does not match the number expected from the count "
                       f"matrix ({count_matrix.shape[1]}).")
    if count_matrix.shape[0] != barcodes.size:
        logger.warning(f"Number of barcodes ({barcodes.size}) in {filename} "
                       f"does not match the number expected from the count "
                       f"matrix ({count_matrix.shape[0]}).")

    return {'matrix': count_matrix,
            'gene_names': feature_names,
            'gene_ids': feature_ids,
            'genomes': genomes,
            'feature_types': feature_types,
            'barcodes': barcodes}
