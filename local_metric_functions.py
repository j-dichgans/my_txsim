import scanpy as sc
import numpy as np
import pandas as pd
from anndata import AnnData
import networkx as nx
import anndata as ad
import matplotlib.pyplot as plt
from tifffile import imread
from scipy.ndimage import gaussian_filter
from scipy.sparse import isspmatrix
from scipy import stats
import itertools
import seaborn as sns
from typing import Tuple
from typing import List

#TODO: fix Warnings in get_knn_mixing_score
#TODO: fix Warnings in get_cells_location

#helper function
def check_crop_exists(x_min: int, x_max: int, y_min: int, y_max: int, image: np.ndarray):
    """Check if crop coordinates exist.
    
    For this, we check if either (x_min, x_max, y_min, y_max) or an image was provided. If not, we raise a ValueError. 

    Parameters
    ----------
    x_min: int, x_max: int, y_min: int, y_max: int
        crop coordinates
    image: np.ndarray

    Returns
    -------
    if no ValueError was raised, returns range  
    """
    if (x_min is None or x_max is None or y_min is None or y_max is None) and image is None:
        raise ValueError("please provide an image or crop")         
        
    if x_min is not None and x_max is not None and y_min is not None and y_max is not None:
        range = [[x_min,x_max],[y_min,y_max]]
    
    else:
        range = [[0,image.shape[0]],[0,image.shape[1]]]
        
    return range 


#helper function
def matrix_colorbar_plot(matrix: np.ndarray, title: str, x_min: int, x_max: int, y_min: int, y_max: int, vmin, vmax, 
                         smooth: float = 0, show_ticks: bool = False):
    """Display (smoothed and cropped) matrix as an image with a colorbar and title.
    
    Parameters
    ----------
    matrix: np.ndarray
        data
    title: str
    x_min: int, x_max: int, y_min: int, y_max: int
        crop coordinates
    smooth : float = 0
        sigma parameter of scipy.ndimage.gaussian_filter function
    show_ticks : bool 
        default False, show no ticks or labels
    """
    
    matrix = gaussian_filter(matrix,sigma=smooth)
    fig = plt.figure()
    ax = fig.add_subplot(title = title)
    plot = plt.imshow(matrix, vmin=vmin, vmax=vmax, interpolation='nearest', extent=[x_min, x_max, y_max, y_min])
    fig.colorbar(plot)
    
    if not show_ticks:
        ax.tick_params(which='both', bottom=False, left=False, labelbottom = False, labelleft = False)



def get_negative_marker_dict(adata_sp: AnnData, adata_sc: AnnData, key: str='celltype'):
    """Add dictionary of negative marker genes for different celltypes.
    
    Parameters
    ----------
    adata_sp : AnnData
        Annotated ``AnnData`` object with counts from spatial data
    adata_sc : AnnData
        Annotated ``AnnData`` object with counts scRNAseq data
    key : str
        Celltype key in adata_sp.obs and adata_sc.obs

    Returns
    -------
    neg_marker_dict : dict[string: list]
        Dictionary with celltypes as keys and corresponding negative marker genes as values
    """

    # Set threshold parameters
    min_number_cells=10 # minimum number of cells belonging to a cluster to consider it in the analysis
    max_ratio_cells=0.005 # maximum ratio of cells expressing a marker to call it a negative marker gene-ct pair

    #check that genes in spatial data is subset of genes in single cell data
    adata_sp = adata_sp[:,adata_sp.var_names.isin(adata_sc.var_names)]

    # Subset adata_sc to genes of spatial data
    adata_sc = adata_sc[:,adata_sp.var_names]           
    
    adata_sc.layers["raw"] = adata_sc.X

    # TMP fix for sparse matrices, ideally we don't convert, and instead have calculations for sparse/non-sparse
    if isspmatrix(adata_sc.layers["raw"]):                                  
        adata_sc.layers["raw"] = adata_sc.layers["raw"].toarray()

    # Get cell types that we find in both modalities
    shared_celltypes = adata_sc.obs.loc[adata_sc.obs[key].isin(adata_sp.obs[key]),key].unique()
    
    # Filter cell types by minimum number of cells
    celltype_count_sc = adata_sc.obs[key].value_counts().loc[shared_celltypes]
    celltype_count_sp = adata_sc.obs[key].value_counts().loc[shared_celltypes]   
    ct_filter = (celltype_count_sc >= min_number_cells) & (celltype_count_sp >= min_number_cells)
    celltypes = celltype_count_sc.loc[ct_filter].index.tolist()             
    
    
    # Filter cells to eligible cell types
    adata_sc = adata_sc[adata_sc.obs[key].isin(celltypes)]
    genes = adata_sc.var_names
    
    # Get ratio of positive cells per cell type
    pos_exp_sc = pd.DataFrame(adata_sc.layers["raw"] > 0,columns=adata_sp.var_names)     
    pos_exp_sc['celltype'] = list(adata_sc.obs[key])
    ratio_celltype_sc = pos_exp_sc.groupby('celltype').mean()
    
    # Get gene-cell type pairs with negative marker expression
    neg_marker_mask = np.array(ratio_celltype_sc < max_ratio_cells)
    

    neg_marker_mask = pd.DataFrame(neg_marker_mask, columns = genes, index = ratio_celltype_sc.index)   
    neg_marker_dict = {celltype: list(row.index[row]) for celltype, row in neg_marker_mask.iterrows()}

    return neg_marker_dict



def get_spot_assignment_col(adata_sp: AnnData, adata_sc: AnnData):

    """Add column spot_assignment to adata_sp.uns["spots"].

       spot_assignment is "spot in wrong celltype", if celltype of cell is in the keys of the neg_marker_dict and gene is a negative marker for the celltype. 
       spot_assignment is "unassigned" if the corresponding cell entry is NaN.
       spot_assignment is "no negative marker" if gene is  not a negative marker for any celltype in neg_marker_dict.
       spot_assignment is "spot in correct celltype" otherwise.
      
       Parameters
       ----------
       adata_sp : AnnData
        Annotated ``AnnData`` object with counts from spatial data
       adata_sc : AnnData
        Annotated ``AnnData`` object with counts scRNAseq data
    """ 
    #check that genes in spatial data is subset of genes in single cell data
    adata_sp = adata_sp[:,adata_sp.var_names.isin(adata_sc.var_names)]

    df = adata_sp.uns["spots"]  
   
    neg_marker_dict = get_negative_marker_dict(adata_sp, adata_sc)

    celltypes = list(neg_marker_dict.keys())   

    df["spot_assignment"] = "spot in correct celltype"

    all_neg_markers = set(itertools.chain(*neg_marker_dict.values()))
    #gene is not a negative marker for any celltype in neg_marker_dict and celltype is not in celltypes 
    df.loc[(~df["Gene"].isin(all_neg_markers))& (~df["celltype"].isin(celltypes)), "spot_assignment"] = "no negative marker"  

    for ct in celltypes:
      df.loc[(df["Gene"].isin(neg_marker_dict[ct])) & (df["celltype"] == ct), 'spot_assignment'] = "spot in wrong celltype" 

    df.loc[df["cell"]!=df["cell"],"spot_assignment"] = "unassigned"   #check for NaNs



def get_cells_location(adata_sp: AnnData, adata_sc: AnnData):
    """Add x,y coordinate columns of cells to adata_sp.obs.

        Parameters
        ----------
        adata_sp : AnnData
            Annotated ``AnnData`` object with counts from spatial data
        adata_sc : AnnData
            Annotated ``AnnData`` object with counts scRNAseq data
    """

    get_spot_assignment_col(adata_sp,adata_sc)
    spots = adata_sp.uns["spots"]
    df_cells = spots.loc[spots["spot_assignment"]!="unassigned"]      
    df_cells = df_cells.groupby(["cell"])[["x","y"]].mean()
    df_cells = df_cells.reset_index().rename(columns={'cell':'cell_id'})
    adata_sp.obs = pd.merge(df_cells,adata_sp.obs,left_on="cell_id",right_on="cell_id",how="inner")



def get_knn_mixing_score(adata_st: AnnData, adata_sc: AnnData, obs_key: str = "celltype", k: int = 45,ct_filter_factor: float = 2):
    """Get column in adata_sp.obs with knn mixing score.

    For this we concatenate the spatial and single cell datasets, compute the neighborsgraph for eligible celltypes, get the expected value for the
    modality ratio, compute the actual ratio for each cell and assign a the knn mixing score.

    Parameters
    ----------
    adata_sp : AnnData
        Annotated ``AnnData`` object with counts from spatial data
    adata_sc : AnnData
        Annotated ``AnnData`` object with counts scRNAseq data
    """
    adata_st.obs["modality"] = "spatial"
    adata_sc.obs["modality"] = "sc"
    adata = ad.concat([adata_st, adata_sc], join='inner')  
    adata.X = adata.X.astype(np.float32)
    adata.obs_names = adata.obs_names.astype(str) + "_" + adata.obs["modality"].astype(str)
    adata.obs = adata.obs.reset_index()

    # Set counts to log norm data
    adata.X = adata.layers["lognorm"]
    
    # Calculate PCA (Note: we could also think about pca per cell type...)
    assert (adata.obsm is None) or ('X_pca' not in adata.obsm), "PCA already exists."
    sc.tl.pca(adata)
    
    # get cell type groups
    sc_cts = set(adata_sc.obs["celltype"].cat.categories)
    st_cts = set(adata_st.obs["celltype"].cat.categories)
    shared_cts = list(sc_cts.intersection(st_cts))         

    # Get ratio per shared cell type
    df = pd.DataFrame(columns=["index","knn mixing score"])
    for ct in shared_cts:
        enough_cells = (adata.obs.loc[adata.obs[obs_key]==ct,"modality"].value_counts() > (ct_filter_factor * k)).all()     
        if enough_cells:
            a = adata[adata.obs[obs_key]==ct]
            exp_val = (a.obs.loc[a.obs["modality"]=="sc"].shape[0])/a.obs.shape[0]  
            sc.pp.neighbors(a,n_neighbors=k)
            G = nx.Graph(incoming_graph_data=a.obsp["connectivities"])
            nx.set_node_attributes(G, {i:a.obs["modality"].values[i] for i in range(G.number_of_nodes())}, "modality")   

            ct_df = np.zeros(a.obs.shape[0])
            f = np.vectorize(lambda x: x/exp_val if x>=0 and x<=exp_val else x/(exp_val-1)+1/(1-exp_val))
            i = 0
            for cell in G.nodes():
                ct_df[i] = sum(1 for neighbor in G.neighbors(cell) if G.nodes[cell]["modality"]=="sc")  #number_modality_sc
                ct_df[i] = ct_df[i]/G.degree(cell)      #ratio: number modality sc / total cells
                i += 1 

            df_t = a.obs[a.obs['index'].str.endswith('spatial')]    	    #TODO: früher subsetten
            df_t["index"] = df_t["index"].apply(lambda s: s.rsplit('_',1)[0])
            ct_df = ct_df[a.obs['index'].str.endswith('spatial')]
            
            daf = pd.DataFrame({"index": df_t["index"], "knn mixing score":f(ct_df)})
            df = pd.concat([df,daf])

    adata_st.obs = pd.merge(adata_st.obs.reset_index(),df,left_on="index",right_on="index",how="left")   



def get_wrong_spot_ratio(adata_sp : AnnData, x_min: int, x_max: int, y_min: int, y_max: int, image: np.ndarray, bins):
    """Get ratio array of spots in wrong celltype.

    Parameters
    ----------
    adata_sp : AnnData
        Annotated ``AnnData`` object with counts from spatial data
    x_min : int, x_max : int, y_min : int, y_max : int 
        crop coordinates
    image : NDArray
        read from image of dapi stained cell-nuclei
    bins : int or array_like or [int, int] or [array, array]
        The bin specification:
        If int, the number of bins for the two dimensions (nx=ny=bins).
        If array_like, the bin edges for the two dimensions (x_edges=y_edges=bins).
        If [int, int], the number of bins in each dimension (nx, ny = bins).
        If [array, array], the bin edges in each dimension (x_edges, y_edges = bins).
        A combination [int, array] or [array, int], where int is the number of bins and array is the bin edges.
    Returns
    -------
    H : array of floats
        ratio: spots in wrong celltype over spots in correct celltype or no negative marker spots
    range : range of binning 
    """
    df = adata_sp.uns["spots"]
    np.seterr(invalid='ignore')

    range = check_crop_exists(x_min,x_max,y_min,y_max,image)

    true_spots = df.loc[(df['spot_assignment'] == "spot in correct celltype") | (df['spot_assignment'] == "no negative marker")]
    H_t = np.histogram2d(true_spots['x'], true_spots['y'], bins, range)[0].T

    spots_in_wrong_ct = df.loc[df['spot_assignment'] == "spot in wrong celltype"]
    H_w = np.histogram2d(spots_in_wrong_ct['x'], spots_in_wrong_ct['y'], bins, range)[0].T

    H = H_w/(H_t+H_w)       #ignore "Unassigned" spots in density calculation, since it could be both assigned correctly or wrongly
    H[np.isnan(H)] = 0      #no negative marker wrongly assigned where no spots

 
    return H,range
    


def get_spot_density(adata_sp : AnnData, spot_types: List[str], x_min: int, x_max: int, y_min: int, y_max: int, image: np.ndarray, bins):
    """Get density of specified spot types ("spot in wrong celltype", "spot in correct celltype", "unassigned", "no negative marker")

    Parameters
    ----------
    adata_sp : AnnData
        Annotated ``AnnData`` object with counts from spatial data
    spot_types: list['str']
        specify, which spot types ("spot in wrong celltype", "spot in correct celltype", "unassigned", "no negative marker") should be included in calculation
    x_min : int, x_max : int, y_min : int, y_max : int 
        crop coordinates
    image : NDArray
        read from image of dapi stained cell-nuclei
    bins : int or array_like or [int, int] or [array, array]
        The bin specification:
        If int, the number of bins for the two dimensions (nx=ny=bins).
        If array_like, the bin edges for the two dimensions (x_edges=y_edges=bins).
        If [int, int], the number of bins in each dimension (nx, ny = bins).
        If [array, array], the bin edges in each dimension (x_edges, y_edges = bins).
        A combination [int, array] or [array, int], where int is the number of bins and array is the bin edges.
    Returns
    -------
    H : array of floats
        density of specified spottypes reads per bin
    range : range of binning 
    """
    df = adata_sp.uns["spots"]

    range = check_crop_exists(x_min,x_max,y_min,y_max,image)
    
    spots = df.loc[df['spot_assignment'].isin(spot_types)]          
    H = np.histogram2d(spots['x'], spots['y'], bins, range)[0].T

    return H, range


def get_cell_density(adata_sp: AnnData, x_min: int, x_max: int, y_min: int, y_max: int, image: np.ndarray, bins):
    """Get cell density.

    Parameters
    ----------
    adata_sp: AnnData
        Annotated ``AnnData`` object with counts from spatial data
    x_min : int, x_max : int, y_min : int, y_max : int 
        crop coordinates
    image : NDArray
        read from image of dapi stained cell-nuclei
    bins : int or array_like or [int, int] or [array, array]
        The bin specification:
        If int, the number of bins for the two dimensions (nx=ny=bins).
        If array_like, the bin edges for the two dimensions (x_edges=y_edges=bins).
        If [int, int], the number of bins in each dimension (nx, ny = bins).
        If [array, array], the bin edges in each dimension (x_edges, y_edges = bins).
        A combination [int, array] or [array, int], where int is the number of bins and array is the bin edges.
    Returns
    -------
    H : array of floats
        density of cells per bin
    range : range of binning 
    """
     
    df_cells = adata_sp.obs 
    range = check_crop_exists(x_min,x_max,y_min,y_max,image)
    
    H = np.histogram2d(df_cells['x'], df_cells['y'], bins, range)[0].T

    return H, range



def get_celltype_density(adata_sp: AnnData, celltype: str, x_min: int, x_max: int, y_min: int, y_max: int, image: np.ndarray, bins):
    """Get celltype density.

    Parameters
    ----------
    adata_sp: AnnData
        Annotated ``AnnData`` object with counts from spatial data
    celltype: str
        celltype
    x_min : int, x_max : int, y_min : int, y_max : int 
        crop coordinates
    image : NDArray
        read from image of dapi stained cell-nuclei
    bins : int or array_like or [int, int] or [array, array]
        The bin specification:
        If int, the number of bins for the two dimensions (nx=ny=bins).
        If array_like, the bin edges for the two dimensions (x_edges=y_edges=bins).
        If [int, int], the number of bins in each dimension (nx, ny = bins).
        If [array, array], the bin edges in each dimension (x_edges, y_edges = bins).
        A combination [int, array] or [array, int], where int is the number of bins and array is the bin edges.
    Returns
    -------
    H : array of floats
        density of celltype per bin
    range : range of binning 
    """
    df =  adata_sp.obs
    range = check_crop_exists(x_min,x_max,y_min,y_max,image)
    
    H_total = np.histogram2d(df['x'],df['y'], bins, range)[0]

    df = df.loc[df["celltype"]==celltype]

    H_celltype = np.histogram2d(df['x'],df['y'], bins, range)[0]

    H = H_celltype/H_total
    H[np.isnan(H)] = 0     #0 cells therefore also 0% celltype in the respective area
    H = H.T
    
    return H, range



def get_number_of_celltypes(adata_sp: AnnData, x_min: int, x_max: int, y_min: int, y_max: int, image: np.ndarray, bins: Tuple[int,int]):
    """Get number of celltypes
    
    Parameters
    ---------
    adata_sp: AnnData
        Annotated ``AnnData`` object with counts from spatial data
    x_min : int, x_max : int, y_min : int, y_max : int 
        crop coordinates
    image : NDArray
        read from image of dapi stained cell-nuclei
    bins : [int,int]
        the number of bins in each dimension
    Returns
    -------
    H : array of floats
        number of celltypes per bin
    range : range of binning 
    """
    
    range_ = check_crop_exists(x_min,x_max,y_min,y_max,image)
    
    celltypes = adata_sp.obs["celltype"].unique()  
    A = np.zeros((len(celltypes),bins[0],bins[1]))   

    for i in range(len(celltypes)):
        A[i,...] = get_celltype_density(adata_sp, celltypes[i],x_min,x_max,y_min,y_max,image,bins)[0]
    A = sum(A>0)
    
    return A, range_



def get_major_celltype_perc(adata_sp: AnnData, x_min: int, x_max: int, y_min: int, y_max: int, image: np.ndarray, bins: Tuple[int,int]):
    """Get major celltype percentage.
    
    Parameters
    ---------
    adata_sp: AnnData
        Annotated ``AnnData`` object with counts from spatial data
    x_min : int, x_max : int, y_min : int, y_max : int 
        crop coordinates
    image : NDArray
        read from image of dapi stained cell-nuclei
    bins : [int,int]
        the number of bins in each dimension
    Returns
    -------
    H : array of floats
        major celltype percentage
    range : range of binning 
    """
    range_ = check_crop_exists(x_min,x_max,y_min,y_max,image)
    
    celltypes = adata_sp.obs["celltype"].unique()
    A = np.zeros((len(celltypes),bins[0],bins[1]))   
    
    for i in range(len(celltypes)):
        A[i,...] = get_celltype_density(adata_sp, celltypes[i],x_min,x_max,y_min,y_max,image,bins)[0]
    B = np.max(A,axis=0)

    return B, range_



def get_summed_cell_area(adata_sp: AnnData, x_min: int, x_max: int, y_min: int, y_max: int, image: np.ndarray, bins: Tuple[int,int]):
    """Get summed cell area.
    
    Parameters
    ----------
    adata_sp: AnnData
        Annotated ``AnnData`` object with counts from spatial data
    x_min : int, x_max : int, y_min : int, y_max : int 
        crop coordinates
    image : NDArray
        read from image of dapi stained cell-nuclei
    bins : [int,int]
        the number of bins in each dimension
    Returns
    -------
    H : array of floats
        summed cell area
    range : range of binning 
    """
    df = adata_sp.obs
    range = check_crop_exists(x_min,x_max,y_min,y_max,image)
    x_min, x_max, y_min, y_max = np.ravel(range).tolist()

    #filter spots
    df = df.loc[(df['x']>= x_min) & (df['x']<=x_max) & (df['y']>=y_min) & (df['y']<=y_max)]

    bins_x = np.digitize(df['x'], np.linspace(x_min, x_max, bins[0] + 1)) -1
    bins_y = np.digitize(df['y'], np.linspace(y_min, y_max, bins[1] + 1)) -1

    groups = df.groupby([bins_x, bins_y])
    sums = groups['area'].sum()
    summed_cell_area = np.zeros((bins[0],bins[1]))

    for (i,j), value in sums.items():
        summed_cell_area[i,j] = value
    
    return summed_cell_area.T, range



def get_avg_knn_mixing(adata_sp: AnnData, x_min: int, x_max: int, y_min: int, y_max: int, 
                       image: np.ndarray, bins: Tuple[int,int]):
    """Get average knn mixing score.
    
    Parameters
    ----------
    adata_sp: AnnData
        Annotated ``AnnData`` object with counts from spatial data
    x_min : int, x_max : int, y_min : int, y_max : int 
        crop coordinates
    image : NDArray
        read from image of dapi stained cell-nuclei
    bins : [int,int]
        the number of bins in each dimension
    Returns
    -------
    H : array of floats
        average knn mixing score
    range : range of binning 
    """

    df = adata_sp.obs
    range = check_crop_exists(x_min,x_max,y_min,y_max,image)
    x_min, x_max, y_min, y_max = np.ravel(range).tolist()

    #filter spots
    df = df.loc[(df['x']>= x_min) & (df['x']<=x_max) & (df['y']>=y_min) & (df['y']<=y_max)]

    bins_x = np.digitize(df['x'], np.linspace(x_min, x_max, bins[0] + 1)) -1
    bins_y = np.digitize(df['y'], np.linspace(y_min, y_max, bins[1] + 1)) -1

    groups = df.groupby([bins_x, bins_y])
    nanmean = groups['score'].mean()            #?better with np.nanmean()
    average_knn_mixing_score = np.zeros((bins[0],bins[1]))

    for (i,j), value in nanmean.items():
        average_knn_mixing_score[i,j] = value
    
    return average_knn_mixing_score.T, range



def get_correlation_matrices(adata_sp: AnnData, x_min: int, x_max: int, y_min: int, y_max: int, image: np.ndarray, bins: Tuple[int,int], celltype: str):
    """Get pearson and spearman correlation matrices.

    Parameters
    ----------
    adata_sp: AnnData
        Annotated ``AnnData`` object with counts from spatial data
    x_min : int, x_max : int, y_min : int, y_max : int 
        crop coordinates
    image : NDArray
        read from image of dapi stained cell-nuclei
    bins : [int,int]
        the number of bins in each dimension
    
    Returns
    -------
        pearson and spearman correlation matrices, df with measurements
    """

    M_wrong_spot_ratio = get_wrong_spot_ratio(adata_sp,x_min, x_max, y_min, y_max,image,bins)[0]
    M_spot_density = get_spot_density(adata_sp,["spot in wrong celltype"],x_min, x_max, y_min, y_max,image,bins)[0]
    M_cell_density = get_cell_density(adata_sp,x_min, x_max, y_min, y_max,image,bins)[0]
    M_celltype_density = get_celltype_density(adata_sp,celltype, x_min, x_max, y_min, y_max,image,bins)[0] 
    M_number_of_celltypes = get_number_of_celltypes(adata_sp,x_min, x_max, y_min, y_max,image,bins)[0]
    M_major_celltype_perc = get_major_celltype_perc(adata_sp,x_min, x_max, y_min, y_max,image,bins)[0]
    M_summed_cell_area = get_summed_cell_area(adata_sp,x_min, x_max, y_min, y_max,image,bins)[0]
    M_avg_knn_mixing = get_avg_knn_mixing(adata_sp,x_min, x_max, y_min, y_max,image,bins)[0]

    measurements_df = pd.DataFrame({'wrong spot ratio':M_wrong_spot_ratio.flatten(), 'spot density': M_spot_density.flatten(), 
                                'cell density': M_cell_density.flatten(), 'celltype density': M_celltype_density.flatten(), 'number of celltypes': M_number_of_celltypes.flatten(),
                                'major celltype perc': M_major_celltype_perc.flatten(), 'summed cell area': M_summed_cell_area.flatten(),
                                'avg knn mixing': M_avg_knn_mixing.flatten()})
    n = measurements_df.shape[1]
    spearman_corr_matrix = np.zeros((n,n))
    pearson_corr_matrix = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            spearman_corr_matrix[i,j] = stats.spearmanr(measurements_df.iloc[:,i],measurements_df.iloc[:,j]).statistic
            pearson_corr_matrix[i,j] = stats.pearsonr(measurements_df.iloc[:,i],measurements_df.iloc[:,j]).statistic
    
    return pearson_corr_matrix, spearman_corr_matrix, measurements_df




            


