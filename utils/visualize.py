import numpy as np
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.calibration import calibration_curve

sns.set_style("whitegrid")  # style global plus clean


def plot_confusion_matrices(
    y_true_valid,
    y_pred_valid,
    y_true_test,
    y_pred_test,
    class_names,
    normalize: bool = False,
):
    """
    Affiche côte à côte les matrices de confusion (Validation vs Test).

    Args:
        y_true_valid (array-like): vraies étiquettes validation
        y_pred_valid (array-like): prédictions validation
        y_true_test (array-like): vraies étiquettes test
        y_pred_test (array-like): prédictions test
        class_names (list): noms des classes (ex: encoder.classes_)
        normalize (bool): si True, affiche les fréquences (%) au lieu des comptes
    """
    labels = list(range(len(class_names)))

    # Matrices de confusion
    cm_valid = confusion_matrix(y_true_valid, y_pred_valid, labels=labels)
    cm_test = confusion_matrix(y_true_test, y_pred_test, labels=labels)

    if normalize:
        cm_valid = cm_valid.astype("float") / cm_valid.sum(axis=1)[:, np.newaxis]
        cm_test = cm_test.astype("float") / cm_test.sum(axis=1)[:, np.newaxis]
        fmt = ".2f"
    else:
        fmt = "d"

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- Validation ---
    sns.heatmap(
        cm_valid,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False,
        ax=axes[0],
        linewidths=0.5,
        linecolor="gray",
    )
    axes[0].set_title("Validation", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Prédictions")
    axes[0].set_ylabel("Vraies classes")
    plt.setp(axes[0].get_xticklabels(), rotation=45, ha="right")

    # --- Test ---
    sns.heatmap(
        cm_test,
        annot=True,
        fmt=fmt,
        cmap="Greens",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False,
        ax=axes[1],
        linewidths=0.5,
        linecolor="gray",
    )
    axes[1].set_title("Test", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Prédictions")
    axes[1].set_ylabel("")
    plt.setp(axes[1].get_xticklabels(), rotation=45, ha="right")

    # --- Titre global ---
    plt.suptitle(
        " Matrices de confusion - Validation vs Test",
        fontsize=18,
        fontweight="bold",
        y=1.05,
    )
    plt.tight_layout()
    plt.show()


# -------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------#
def plot_max_proba_distribution(y_proba_valid, y_proba_test):
    """
    Visualise la distribution des probabilités maximales (confiance du modèle).

    Objectif:
        - Évaluer si le modèle est généralement confiant (probas proches de 1)
          ou incertain (probas autour de 0.5).
        - Comparer la répartition des confiances entre Validation et Test.

    Args:
        y_proba_valid (array): prédictions de probabilité pour la validation (n_samples x n_classes)
        y_proba_test (array): prédictions de probabilité pour le test (n_samples x n_classes)
    """
    max_proba_valid = np.max(y_proba_valid, axis=1)
    max_proba_test = np.max(y_proba_test, axis=1)

    plt.figure(figsize=(8, 5))
    sns.histplot(
        max_proba_valid,
        bins=20,
        kde=True,
        color="royalblue",
        label="Validation",
        stat="density",
        alpha=0.6,
    )
    sns.histplot(
        max_proba_test,
        bins=20,
        kde=True,
        color="darkorange",
        label="Test",
        stat="density",
        alpha=0.6,
    )
    plt.title("Distribution des probabilités maximales")
    plt.xlabel("Probabilité prédite max")
    plt.ylabel("Densité")
    plt.legend()
    plt.show()


def plot_classwise_confidence(y_proba_valid, y_proba_test, class_names):
    """
    Visualise la confiance du modèle par classe prédite (via un boxplot).

    Objectif:
        - Identifier les classes pour lesquelles le modèle est très confiant
          (distributions concentrées vers 1.0).
        - Détecter les classes plus ambiguës avec de fortes variabilités.
        - Comparer Validation vs Test pour repérer un éventuel surapprentissage.

    Args:
        y_proba_valid (array): prédictions de probabilité pour la validation (n_samples x n_classes)
        y_proba_test (array): prédictions de probabilité pour le test (n_samples x n_classes)
        class_names (list): noms des classes dans l'ordre encodé
    """
    y_pred_valid = np.argmax(y_proba_valid, axis=1)
    y_pred_test = np.argmax(y_proba_test, axis=1)

    pred_max_proba_valid = np.max(y_proba_valid, axis=1)
    pred_max_proba_test = np.max(y_proba_test, axis=1)

    plt.figure(figsize=(14, 6))
    sns.boxplot(
        x=[class_names[i] for i in y_pred_valid],
        y=pred_max_proba_valid,
        color="royalblue",
        width=0.4,
        fliersize=2,
    )
    sns.boxplot(
        x=[class_names[i] for i in y_pred_test],
        y=pred_max_proba_test,
        color="darkorange",
        width=0.4,
        fliersize=2,
    )
    plt.title("Confiance (probabilité max) par classe prédite")
    plt.xticks(rotation=45)
    plt.ylabel("Probabilité max")
    plt.show()


def plot_calibration_curve(y_true_valid, y_proba_valid, y_true_test, y_proba_test):
    """
    Trace la courbe de calibration (reliability diagram).

    Objectif:
        - Vérifier si les probabilités émises par le modèle sont bien calibrées :
          ex: une proba de 0.8 correspond bien à ~80% de bonnes prédictions.
        - Comparer la calibration entre Validation et Test.

    Args:
        y_true_valid (array): vraies classes pour la validation
        y_proba_valid (array): prédictions de probabilité validation (n_samples x n_classes)
        y_true_test (array): vraies classes pour le test
        y_proba_test (array): prédictions de probabilité test (n_samples x n_classes)
    """
    max_proba_valid = np.max(y_proba_valid, axis=1)
    max_proba_test = np.max(y_proba_test, axis=1)

    y_pred_valid = np.argmax(y_proba_valid, axis=1)
    y_pred_test = np.argmax(y_proba_test, axis=1)

    plt.figure(figsize=(8, 6))
    prob_true_val, prob_pred_val = calibration_curve(
        (y_true_valid == y_pred_valid).astype(int), max_proba_valid, n_bins=10
    )
    prob_true_test, prob_pred_test = calibration_curve(
        (y_true_test == y_pred_test).astype(int), max_proba_test, n_bins=10
    )

    plt.plot(
        prob_pred_val, prob_true_val, marker="o", label="Validation", color="royalblue"
    )
    plt.plot(
        prob_pred_test, prob_true_test, marker="s", label="Test", color="darkorange"
    )
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Idéal")
    plt.title("Diagramme de calibration (multiclass via proba max)")
    plt.xlabel("Probabilité prédite")
    plt.ylabel("Fréquence observée")
    plt.legend()
    plt.show()


# -------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------#
def plot_max_proba_distribution_px(y_proba_valid, y_proba_test):
    """
    Visualise la distribution des probabilités maximales avec Plotly Express.

    Objectif:
        - Comparer les distributions de confiance (Validation vs Test).
        - Graph interactif (zoom, hover, légendes dynamiques).
    """
    max_proba_valid = np.max(y_proba_valid, axis=1)
    max_proba_test = np.max(y_proba_test, axis=1)

    data = [(p, "Validation") for p in max_proba_valid] + [
        (p, "Test") for p in max_proba_test
    ]
    proba, dataset = zip(*data)

    fig = px.histogram(
        x=proba,
        color=dataset,
        nbins=20,
        marginal="box",
        opacity=0.7,
        barmode="overlay",
        labels={"x": "Probabilité prédite max", "color": "Dataset"},
        title="Distribution des probabilités maximales",
    )
    fig.show()


def plot_classwise_confidence_px(y_proba_valid, y_proba_test, class_names):
    """
    Visualise la confiance du modèle par classe prédite (violin plot interractif).

    Objectif:
        - Explorer la variabilité des probas par classe.
        - Comparer Validation vs Test avec un graphe interactif.
    """
    y_pred_valid = np.argmax(y_proba_valid, axis=1)
    y_pred_test = np.argmax(y_proba_test, axis=1)

    pred_max_proba_valid = np.max(y_proba_valid, axis=1)
    pred_max_proba_test = np.max(y_proba_test, axis=1)

    data = [
        (class_names[i], p, "Validation")
        for i, p in zip(y_pred_valid, pred_max_proba_valid)
    ] + [(class_names[i], p, "Test") for i, p in zip(y_pred_test, pred_max_proba_test)]
    classes, probs, dataset = zip(*data)

    fig = px.violin(
        x=classes,
        y=probs,
        color=dataset,
        box=True,
        points="all",
        labels={"x": "Classe prédite", "y": "Probabilité max", "color": "Dataset"},
        title="Confiance (probabilité max) par classe prédite",
    )
    fig.update_xaxes(tickangle=45)
    fig.show()


def plot_calibration_curve_px(y_true_valid, y_proba_valid, y_true_test, y_proba_test):
    """
    Trace la courbe de calibration avec Plotly Express.

    Objectif:
        - Vérifier l’alignement entre proba prédite et fréquence observée.
        - Comparer Validation vs Test avec une visualisation interactive.
    """
    max_proba_valid = np.max(y_proba_valid, axis=1)
    max_proba_test = np.max(y_proba_test, axis=1)

    y_pred_valid = np.argmax(y_proba_valid, axis=1)
    y_pred_test = np.argmax(y_proba_test, axis=1)

    prob_true_val, prob_pred_val = calibration_curve(
        (y_true_valid == y_pred_valid).astype(int), max_proba_valid, n_bins=10
    )
    prob_true_test, prob_pred_test = calibration_curve(
        (y_true_test == y_pred_test).astype(int), max_proba_test, n_bins=10
    )

    data = [(x, y, "Validation") for x, y in zip(prob_pred_val, prob_true_val)] + [
        (x, y, "Test") for x, y in zip(prob_pred_test, prob_true_test)
    ]
    prob_pred, prob_true, dataset = zip(*data)

    fig = px.line(
        x=prob_pred,
        y=prob_true,
        color=dataset,
        markers=True,
        labels={"x": "Probabilité prédite", "y": "Fréquence observée"},
        title="Diagramme de calibration (multiclass via proba max)",
    )
    fig.add_scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        line=dict(dash="dash", color="gray"),
        name="Idéal",
    )
    fig.show()


# -------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------#
def plot_box_confidence_overlay(y_proba_valid, y_proba_test, class_names):
    """
    Boxplot superposé des probabilités maximales par classe prédite.
    (Validation et Test affichés l'un sur l'autre)
    """
    y_pred_valid = np.argmax(y_proba_valid, axis=1)
    y_pred_test = np.argmax(y_proba_test, axis=1)

    pred_max_proba_valid = np.max(y_proba_valid, axis=1)
    pred_max_proba_test = np.max(y_proba_test, axis=1)

    plt.figure(figsize=(14, 6))
    sns.boxplot(
        x=[class_names[i] for i in y_pred_valid],
        y=pred_max_proba_valid,
        color="royalblue",
        width=0.4,
        fliersize=2,
    )
    sns.boxplot(
        x=[class_names[i] for i in y_pred_test],
        y=pred_max_proba_test,
        color="darkorange",
        width=0.4,
        fliersize=2,
    )
    plt.title(
        "Confiance (probabilité max) par classe prédite (Overlay)",
        fontsize=14,
        fontweight="bold",
    )
    plt.xticks(rotation=45)
    plt.ylabel("Probabilité max")
    plt.show()


def plot_box_confidence_split(y_proba_valid, y_proba_test, class_names):
    """
    Boxplot côte à côte (split) des probabilités maximales par classe prédite.
    (Validation vs Test séparés par la couleur et la légende)
    """
    y_pred_valid = np.argmax(y_proba_valid, axis=1)
    y_pred_test = np.argmax(y_proba_test, axis=1)

    pred_max_proba_valid = np.max(y_proba_valid, axis=1)
    pred_max_proba_test = np.max(y_proba_test, axis=1)

    # Préparation des données pour seaborn
    data = [
        (class_names[i], p, "Validation")
        for i, p in zip(y_pred_valid, pred_max_proba_valid)
    ] + [(class_names[i], p, "Test") for i, p in zip(y_pred_test, pred_max_proba_test)]
    classes, probs, dataset = zip(*data)

    plt.figure(figsize=(14, 6))
    sns.boxplot(
        x=classes,
        y=probs,
        hue=dataset,
        palette={"Validation": "royalblue", "Test": "darkorange"},
    )
    plt.title(
        "Confiance (probabilité max) par classe prédite (Split)",
        fontsize=14,
        fontweight="bold",
    )
    plt.xticks(rotation=45)
    plt.ylabel("Probabilité max")
    plt.legend(title="Dataset")
    plt.show()
