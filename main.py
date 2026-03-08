import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyrepo_mcda import normalizations as norms
from pyrepo_mcda.additions import rank_preferences
from pyrepo_mcda import weighting_methods as mcda_weights

from pyrepo_mcda.mcda_methods import TOPSIS
from pyrepo_mcda import correlations as corrs
from pyrepo_mcda import distance_metrics as dists

import seaborn as sns
import itertools
from adjustText import adjust_text

# Entropy weighting
def entropy(matrix):
    """
    Calculate criteria weights using objective Entropy weighting method.

    Parameters
    -----------
        matrix : ndarray
            Decision matrix with performance values of m alternatives and n criteria.

    Returns
    --------
        ndarray
            Vector of criteria weights.

    Examples
    ----------
    >>> weights = entropy_weighting(matrix)
    """
    # normalize the decision matrix with `sum_normalization` method from `normalizations` as for profit criteria
    types = np.ones(np.shape(matrix)[1])
    pij = norms.sum_normalization(matrix, types)
    # Transform negative values in decision matrix `matrix` to positive values
    pij = np.abs(pij)
    m, n = np.shape(pij)
    H = np.zeros((m, n))

    # Calculate entropy
    for j, i in itertools.product(range(n), range(m)):
        if pij[i, j]:
            H[i, j] = pij[i, j] * np.log(pij[i, j])

    h = np.sum(H, axis = 0) * (-1 * ((np.log(m)) ** (-1)))

    return h

# heat maps with correlations
def draw_heatmap(df_new_heatmap, title):
    """
    Visualization method to display heatmap with correlations of compared rankings generated using different methods
    
    Parameters
    ----------
        data : DataFrame
            DataFrame with correlation values between compared rankings
        title : str
            title of chart containing name of used correlation coefficient
    Examples
    ---------
    >>> draw_heatmap(df_new_heatmap, title)
    """
    plt.figure(figsize = (12, 8))
    sns.set(font_scale = 1.7)
    heatmap = sns.heatmap(df_new_heatmap, annot=True, fmt=".4f", cmap="RdYlGn",
                          linewidth=0.5, linecolor='w')
    plt.yticks(va="center")
    plt.xlabel('Methods')
    plt.title('Correlation: ' + title)
    plt.tight_layout()
    title = title.replace("$", "")
    plt.savefig('./results/' + 'correlations_' + title + '.pdf')
    plt.show()

# Create dictionary class
class Create_dictionary(dict):
  
    # __init__ function
    def __init__(self):
        self = dict()
          
    # Function to add key:value
    def add(self, key, value):
        self[key] = value




def main():

    # Load decision matrix with performance values
    # base dataset
    dataset = pd.read_csv('offshore windfarms assessment dataset.csv', index_col='Ai')
    criteria_names = [r'$C_{' + str(i) + '}$' for i in range(1, dataset.shape[1] + 1)]
    dataset.columns = criteria_names

    df1 = dataset.iloc[:len(dataset) - 1, :]
    types = dataset.iloc[len(dataset) - 1, :].to_numpy()
    

    # full dataset
    # df1

    matrix = df1.to_numpy()
    # equal weights
    weights = mcda_weights.equal_weighting(matrix)
    names = [r'$A_{' + str(i) + '}$' for i in range(1, df1.shape[0] + 1)]

    results_pref = pd.DataFrame(index=names)
    results_rank = pd.DataFrame(index=names)

    topsis = TOPSIS(normalization_method=norms.minmax_normalization)
    pref_full = topsis(matrix, weights, types)
    results_pref['TOPSIS full dataset'] = pref_full
    rank_full = rank_preferences(pref_full, reverse=True)
    results_rank['TOPSIS full dataset'] = rank_full


    # criteria significance
    crit_sign = pd.DataFrame()
    # CV
    std = np.std(matrix, axis = 0, ddof = 1)
    avg = np.mean(matrix, axis = 0)

    # Calculated CV
    cv = std / avg
    q1 = np.quantile(cv, 0.25)

    crit_sign['CV'] = cv

    ind = list(np.where(cv < q1)[0])

    liczba_zmian_pozycji = []
    # ==================================================================
    # Scenario 1
    # We exclude < q1 from CV
    # df
    df_removed = dataset.drop(dataset.columns[ind], axis=1)
    df2 = df_removed.iloc[:len(df_removed) - 1, :]

    types = df_removed.iloc[len(df_removed) - 1, :].to_numpy()
    matrix = df2.to_numpy()

    # equal weights
    weights = mcda_weights.equal_weighting(matrix)

    topsis = TOPSIS(normalization_method=norms.minmax_normalization)
    pref_t = topsis(matrix, weights, types)
    results_pref['TOPSIS CV q1'] = pref_t
    rank_t = rank_preferences(pref_t, reverse=True)
    results_rank['TOPSIS CV q1'] = rank_t
    liczba_zmian_pozycji.append(np.sum(np.abs(rank_full - rank_t)))


    # ==================================================================
    # Scenario 2
    # We exclude > q3 from Entropy
    # df
    matrix = df1.to_numpy()
    # Calculated Entropy
    entropy_metric = entropy(matrix)
    crit_sign['Entropy'] = entropy_metric
    
    entropy_q3 = np.quantile(entropy_metric, 0.75)
    ind3 = np.where(entropy_metric > entropy_q3)[0]
    
    df_removed = dataset.drop(dataset.columns[ind3], axis=1)
    
    df4 = df_removed.iloc[:len(df_removed) - 1, :]

    types = df_removed.iloc[len(df_removed) - 1, :].to_numpy()
    matrix = df4.to_numpy()

    # equal weights
    weights = mcda_weights.equal_weighting(matrix)

    topsis = TOPSIS(normalization_method=norms.minmax_normalization)
    pref_t = topsis(matrix, weights, types)
    results_pref['TOPSIS Entropy q3'] = pref_t
    rank_t = rank_preferences(pref_t, reverse=True)
    results_rank['TOPSIS Entropy q3'] = rank_t
    liczba_zmian_pozycji.append(np.sum(np.abs(rank_full - rank_t)))

    results_rank.to_csv('results/results.csv')
    results_pref.to_csv('results/results_pref.csv')

    # ===================================================================
    # Correlation
    method_types = list(results_rank.columns)
    dict_new_heatmap_rw = Create_dictionary()
    dict_new_heatmap_rs = Create_dictionary()
    for el in method_types:
        dict_new_heatmap_rw.add(el, [])
        dict_new_heatmap_rs.add(el, [])


    # heatmaps for correlations coefficients
    for i, j in [(i, j) for i in method_types[::-1] for j in method_types]:
        dict_new_heatmap_rw[j].append(corrs.weighted_spearman(results_rank[i], results_rank[j]))
        dict_new_heatmap_rs[j].append(corrs.spearman(results_rank[i], results_rank[j]))

    df_new_heatmap_rw = pd.DataFrame(dict_new_heatmap_rw, index = method_types[::-1])
    df_new_heatmap_rw.columns = method_types

    df_new_heatmap_rs = pd.DataFrame(dict_new_heatmap_rs, index = method_types[::-1])
    df_new_heatmap_rs.columns = method_types

    # correlation matrix with rw coefficient
    draw_heatmap(df_new_heatmap_rw, r'$r_w$')
    draw_heatmap(df_new_heatmap_rs, r'$r_s$')

    print("SHIFTS NUMBER")
    print(liczba_zmian_pozycji)
    print("EUCLIDEAN DISTANCE")
    print(np.round(dists.euclidean(pref_full, pref_t), 4))

    
    # =====================================================================
    # Sensitivity analysis for CV
    # percentiles
    # matching coefficient as percentile for CV
    print("Sensitivity analysis CV")
    results_sens = pd.DataFrame()
    preferences_sens = pd.DataFrame()
    
    percentiles = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

    tabela = pd.DataFrame(index=percentiles, columns=['Parameter',
                                   'Criteria excluded',
                                   'Number of criteria excluded',
                                   '% of criteria excluded',
                                   'Shifts in rankings',
                                   'Euclidean distance',
                                   'Correlation'
                                   ])

    for i, per in enumerate(percentiles):
        
        p = np.percentile(cv, per)
        ind_rem = np.where(cv < p)[0]
        print('CRITERIA excluded after CV p = ', per)
        print(list(dataset.columns[ind_rem]))
        print((len(list(dataset.columns[ind_rem]))*100)/len(list(dataset.columns)), ' %')
        df_removed = dataset.drop(dataset.columns[ind_rem], axis=1)
        
        df_rem = df_removed.iloc[:len(df_removed) - 1, :]

        types = df_removed.iloc[len(df_removed) - 1, :].to_numpy()
        matrix = df_rem.to_numpy()

        # equal weights
        weights = mcda_weights.equal_weighting(matrix)

        topsis = TOPSIS(normalization_method=norms.minmax_normalization)
        pref_t = topsis(matrix, weights, types)
        preferences_sens[str(per)] = pref_t
        rank_t = rank_preferences(pref_t, reverse=True)
        results_sens[str(per)] = rank_t
        print("SHIFTS NUMBER: ", np.sum(np.abs(rank_full - rank_t)))
        print("EUCLIDEAN DISTANCE")
        print(np.round(dists.euclidean(pref_full, pref_t), 4))
        tabela.iloc[i, 0] = per
        tabela.iloc[i, 1] = list(dataset.columns[ind_rem])
        tabela.iloc[i, 2] = len(list(dataset.columns[ind_rem]))
        tabela.iloc[i, 3] = (len(list(dataset.columns[ind_rem]))*100) / len(list(dataset.columns))
        tabela.iloc[i, 4] = np.sum(np.abs(rank_full - rank_t))
        tabela.iloc[i, 5] = np.round(dists.euclidean(pref_full, pref_t), 4)
        tabela.iloc[i, 6] = np.round(corrs.spearman(rank_full, rank_t), 4)

    tabela.to_csv('results/TABLE_CV.csv')
    results_sens.to_csv('results/sens_cv.csv')

    # ----------------------------------------------------------------------
    # PLOTS
    # ----------------------------------------------------------------------
    # plot for CV ranks
    # plot results of analysis with sustainabiblity coefficient modification
    ticks = np.arange(1, matrix.shape[0] + 1)

    x1 = np.arange(0, len(percentiles))

    plt.figure(figsize = (10, 6))
    for i in range(results_sens.shape[0]):
        plt.plot(x1, results_sens.iloc[i, :], '.-', linewidth = 2)
        ax = plt.gca()
        y_min, y_max = ax.get_ylim()
        x_min, x_max = ax.get_xlim()
        plt.annotate(names[i], (x_max - 0.3, results_sens.iloc[i, -1]),
                        fontsize = 16, #style='italic',
                        horizontalalignment='left')

    plt.xlabel("% of criteria removed", fontsize = 16)
    plt.ylabel("Rank", fontsize = 16)
    plt.title('Criteria significance measured with CV', fontsize = 16)
    plt.xticks(x1, np.round(percentiles, 2), fontsize = 16)
    plt.yticks(ticks, fontsize = 16)
    plt.xlim(x_min - 0.2, x_max + 0.5)
    plt.gca().invert_yaxis()
    
    plt.grid(True, linestyle = '--')
    
    plt.tight_layout()
    plt.savefig('./results/rankings_sens_cv' + '.pdf')
    plt.show()

    # ***********************************************************************
    # Closeness Coefficient TOPSIS for CV
    # plot for CV preferences TOPSIS
    # plot results of analysis with sustainabiblity coefficient modification
    ticks = np.arange(0.30, 0.70, 0.05)

    x1 = np.arange(0, len(percentiles))

    texts = []
    plt.figure(figsize=(10, 6))

    for i in range(preferences_sens.shape[0]):
        plt.plot(x1, preferences_sens.iloc[i, :], '.-', linewidth=2)
        texts.append(
            plt.text(
                x1[-1] + 0.05,
                preferences_sens.iloc[i, -1],
                names[i],
                fontsize=16,
                va='center'
            )
        )

    adjust_text(texts, arrowprops=dict(arrowstyle='-', lw=0.5))

    plt.xlabel("% of criteria removed", fontsize=16)
    plt.ylabel("TOPSIS Closeness Coefficient", fontsize=16)
    plt.title('Criteria significance measured with CV', fontsize = 16)
    plt.xticks(x1, np.round(percentiles, 2), fontsize=16)
    plt.yticks(ticks, fontsize=16)
    plt.xlim(x_min - 0.2, x_max + 0.5)
    plt.ylim(0.25, 0.7)
    plt.grid(True, linestyle='--')
    plt.tight_layout()
    plt.savefig('./results/preferences_sens_cv.pdf')
    plt.show()

    
    print("Sensitivity analysis Entropy")
    results_sens = pd.DataFrame()
    preferences_sens = pd.DataFrame()
    
    percentiles = [100, 95, 90, 85, 80, 75, 70, 65, 60, 55, 50]

    tabela = pd.DataFrame(index=percentiles, columns=['Parameter',
                                   'Criteria excluded',
                                   'Number of criteria excluded',
                                   '% of criteria excluded',
                                   'Shifts in rankings',
                                   'Euclidean distance',
                                   'Correlation'
                                   ])

    for i, per in enumerate(percentiles):
        
        p = np.percentile(entropy_metric, per)
        ind_rem = np.where(entropy_metric > p)[0]
        print('CRITERIA usuniete po Entropy p = ', per)
        print(list(dataset.columns[ind_rem]))
        print((len(list(dataset.columns[ind_rem]))*100) / len(list(dataset.columns)), ' %')
        df_removed = dataset.drop(dataset.columns[ind_rem], axis=1)
        df_rem = df_removed.iloc[:len(df_removed) - 1, :]

        types = df_removed.iloc[len(df_removed) - 1, :].to_numpy()
        matrix = df_rem.to_numpy()

        # equal weights
        weights = mcda_weights.equal_weighting(matrix)

        topsis = TOPSIS(normalization_method=norms.minmax_normalization)
        pref_t = topsis(matrix, weights, types)
        preferences_sens[str(per)] = pref_t
        rank_t = rank_preferences(pref_t, reverse=True)
        results_sens[str(per)] = rank_t
        print("SHIFTS NUMBER: ", np.sum(np.abs(rank_full - rank_t)))
        print("EUCLIDEAN DISTANCE")
        print(np.round(dists.euclidean(pref_full, pref_t), 4))
        tabela.iloc[i, 0] = per
        tabela.iloc[i, 1] = list(dataset.columns[ind_rem])
        tabela.iloc[i, 2] = len(list(dataset.columns[ind_rem]))
        tabela.iloc[i, 3] = (len(list(dataset.columns[ind_rem]))*100) / len(list(dataset.columns))
        tabela.iloc[i, 4] = np.sum(np.abs(rank_full - rank_t))
        tabela.iloc[i, 5] = np.round(dists.euclidean(pref_full, pref_t), 4)
        tabela.iloc[i, 6] = np.round(corrs.spearman(rank_full, rank_t), 4)

    tabela.to_csv('results/TABLE_ENTROPY.csv')
    results_sens.to_csv('results/sens_entropy.csv')
    crit_sign.to_csv('results/crieria_significance.csv')

    # ---------------------------------------------------------------------
    # PLOTS
    # ---------------------------------------------------------------------
    # plot
    # plot results of analysis with sustainabiblity coefficient modification
    ticks = np.arange(1, matrix.shape[0] + 1)
    percentiles_reversed = [100 - perc for perc in percentiles]

    x1 = np.arange(0, len(percentiles_reversed))

    plt.figure(figsize = (10, 6))
    for i in range(results_sens.shape[0]):
        plt.plot(x1, results_sens.iloc[i, :], '.-', linewidth = 2)
        ax = plt.gca()
        y_min, y_max = ax.get_ylim()
        x_min, x_max = ax.get_xlim()
        plt.annotate(names[i], (x_max - 0.3, results_sens.iloc[i, -1]),
                        fontsize = 16, #style='italic',
                        horizontalalignment='left')

    plt.xlabel("% of criteria removed", fontsize = 16)
    plt.ylabel("Rank", fontsize = 16)
    plt.title('Criteria significance measured with Entropy', fontsize = 16)
    plt.xticks(x1, np.round(percentiles_reversed, 2), fontsize = 16)
    plt.yticks(ticks, fontsize = 16)
    plt.xlim(x_min - 0.2, x_max + 0.5)
    plt.gca().invert_yaxis()
    
    plt.grid(True, linestyle = '--')
    
    plt.tight_layout()
    plt.savefig('./results/rankings_sens_entropy' + '.pdf')
    plt.show()

    # ******************************************************************
    # Closeness Coefficient TOPSIS for Entropy
    # plot for CV preferences TOPSIS
    # plot results of analysis with sustainabiblity coefficient modification
    ticks = np.arange(0.30, 0.70, 0.05)

    x1 = np.arange(0, len(percentiles_reversed))

    texts = []
    plt.figure(figsize=(10, 6))

    for i in range(preferences_sens.shape[0]):
        plt.plot(x1, preferences_sens.iloc[i, :], '.-', linewidth=2)
        texts.append(
            plt.text(
                x1[-1] + 0.05,
                preferences_sens.iloc[i, -1],
                names[i],
                fontsize=16,
                va='center'
            )
        )

    adjust_text(texts, arrowprops=dict(arrowstyle='-', lw=0.5))

    plt.xlabel("% of criteria removed", fontsize=16)
    plt.ylabel("TOPSIS Closeness Coefficient", fontsize=16)
    plt.title('Criteria significance measured with Entropy', fontsize = 16)
    plt.xticks(x1, np.round(percentiles_reversed, 2), fontsize=16)
    plt.yticks(ticks, fontsize=16)
    plt.xlim(x_min - 0.2, x_max + 0.5)
    plt.ylim(0.25, 0.7)
    plt.grid(True, linestyle='--')
    plt.tight_layout()
    plt.savefig('./results/preferences_sens_entropy.pdf')
    plt.show()

    


if __name__ == '__main__':
    main()