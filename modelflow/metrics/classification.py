import plotly.express as px
import plotly.figure_factory as ff
from sklearn.metrics import confusion_matrix, roc_curve, auc


def plot_roc_curve(m, xs, y, plot=True, skip_thresholds=None):
    pred_y_proba = m.predict_proba(xs)[:, 1]
    fpr, tpr, thresholds = roc_curve(y, pred_y_proba)
    auc_score = auc(fpr, tpr)

    if plot:

        # Reduce memory for plotting
        if skip_thresholds:
            fpr = fpr[::skip_thresholds]
            tpr = tpr[::skip_thresholds]

        fig = px.area(
            x=fpr,
            y=tpr,
            title=f"ROC Curve (AUC={auc_score:.4f})",
            labels=dict(x="False Positive Rate", y="True Positive Rate"),
        )
        fig.add_shape(type="line", line=dict(dash="dash"), x0=0, x1=1, y0=0, y1=1)

        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        fig.update_xaxes(constrain="domain")

        return fig

    return auc_score


def confusion_matrix(m, xs, y, labels=None, normalize=None, colorscale="Blues"):
    pred_y = m.predict(xs)

    cm_labels = sorted(list(y.unique()))
    z = confusion_matrix(y, pred_y, labels=cm_labels, normalize=normalize)

    # invert z idx values
    z = z[::-1]

    x_label = labels if labels else cm_labels
    y_label = x_label[::-1].copy()  # invert idx values of x

    # change each element of z to type string for annotations
    z_text = [[str(y_label) for y_label in x_label] for x_label in z]

    # set up figure
    fig = ff.create_annotated_heatmap(
        z, x=x_label, y=y_label, annotation_text=z_text, colorscale=colorscale
    )

    # add title
    fig.update_layout(
        title_text="<i><b>Confusion matrix</b></i>",
    )

    # add custom xaxis title
    fig.add_annotation(
        dict(
            font=dict(color="black", size=14),
            x=0.5,
            y=-0.15,
            showarrow=False,
            text="Predicted value",
            xref="paper",
            yref="paper",
        )
    )

    # add custom yaxis title
    fig.add_annotation(
        dict(
            font=dict(color="black", size=14),
            x=-0.08,
            y=0.5,
            showarrow=False,
            text="Real value",
            textangle=-90,
            xref="paper",
            yref="paper",
        )
    )

    # adjust margins to make room for yaxis title
    fig.update_layout(margin=dict(t=50, l=200))

    # add colorbar
    fig["data"][0]["showscale"] = True
    return fig
