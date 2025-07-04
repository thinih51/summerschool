import os
import json
from flask import Flask, render_template, url_for, request, jsonify
import pandas as pd
import matplotlib.pyplot as plt
import folium
from folium.plugins import MarkerCluster
from openai import OpenAI
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ── Deine bisherigen Pfade ────────────────────────────────────────────────────
BASE = app.root_path
DATA_DIR = os.path.join(
    BASE, '..', 'Python_Multi-Agent_Augmented_Analytics', 'data')
DATA_PATH = os.path.join(DATA_DIR, 'global-shark-attack.csv')
COORDS_PATH = os.path.join(DATA_DIR, 'location_coordinates.csv')
STATIC_DIR = os.path.join(BASE, 'static')

# ── OpenAI-Key laden ───────────────────────────────────────────────────────────
creds = json.load(
    open(os.path.join(BASE, '..', 'credentials.json'), 'r', encoding='utf-8'))
client = OpenAI(api_key=creds['openai']['api_key'])

# ── Daten-Loader ──────────────────────────────────────────────────────────────


def load_data():
    df = pd.read_csv(DATA_PATH, sep=';', parse_dates=['Date'], dayfirst=True)
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    return df


def load_coords():
    coords = pd.read_csv(COORDS_PATH)
    return coords[['original_Location', 'lat', 'lon']]

# ── Deine bisherigen Plot- und Map-Funktionen ─────────────────────────────────


def create_map(df_merged):
    center = [df_merged['lat'].mean(), df_merged['lon'].mean()]
    m = folium.Map(location=center, zoom_start=2, tiles='CartoDB positron')
    cluster = MarkerCluster().add_to(m)
    for _, row in df_merged.dropna(subset=['lat', 'lon']).iterrows():
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=4,
            popup=row['Location'],
            color='crimson',
            fill=True,
            fill_opacity=0.6
        ).add_to(cluster)
    return m


def plot_attacks_per_year(df):
    yearly = df['Year'].value_counts().sort_index()
    plt.figure(figsize=(6, 4))
    yearly.plot(kind='bar')
    plt.title('Number of Shark Attacks per Year')
    plt.xlabel('Year')
    plt.ylabel('Attacks')
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, 'attacks_per_year.png'))
    plt.close()


def plot_fatal_vs_nonfatal(df):
    counts = df['Fatal'].fillna('N').map(
        lambda x: 'Fatal' if str(x).upper() == 'Y' else 'Non-fatal')
    summary = counts.value_counts()
    plt.figure(figsize=(6, 4))
    summary.plot(kind='pie', autopct='%1.1f%%', startangle=90)
    plt.title('Fatal vs. Non-fatal Shark Attacks')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, 'fatal_vs_nonfatal.png'))
    plt.close()


def plot_top_countries(df):
    top = df['Country'].value_counts().head(8)
    plt.figure(figsize=(6, 4))
    top.plot(kind='barh')
    plt.title('Top 8 Countries by Number of Attacks')
    plt.xlabel('Attacks')
    plt.ylabel('Country')
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, 'top_countries.png'))
    plt.close()


def plot_age_distribution(df):
    ages = df['Age'].dropna()
    plt.figure(figsize=(6, 4))
    plt.hist(ages, bins=range(0, 80, 5), edgecolor='black')
    plt.title('Age Distribution of Victims')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, 'age_distribution.png'))
    plt.close()


# ── Karte einmalig speichern ───────────────────────────────────────────────────
df = load_data()
coords = load_coords()
df_merged = pd.merge(df, coords,
                     how='left',
                     left_on='Location',
                     right_on='original_Location')
m = create_map(df_merged)
m.save(os.path.join(STATIC_DIR, 'map.html'))

# ── Funktion für Function-Calling ─────────────────────────────────────────────


def generate_chart(chart_type: str, x: str, y: str = None, bins: int = None):
    df_local = load_data()
    filename = f"generated_{chart_type}.png"
    path = os.path.join(STATIC_DIR, filename)
    plt.figure(figsize=(6, 4))
    if chart_type in ('bar', 'line'):
        if y:
            df_local.plot(kind=chart_type, x=x, y=y, legend=False)
        else:
            df_local[x].value_counts().sort_index().plot(kind=chart_type)
    elif chart_type == 'pie':
        df_local[x].value_counts().head(10).plot(kind='pie', autopct='%1.1f%%')
    elif chart_type == 'hist':
        df_local[x].dropna().plot(
            kind='hist', bins=bins or 10, edgecolor='black')
    else:
        raise ValueError(f"Unknown chart_type {chart_type}")
    plt.title(f"{chart_type.title()} of {x}" + (f" vs {y}" if y else ""))
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return {'chart_url': url_for('static', filename=filename)}


tools = [{
    "type": "function",
    "function": {
        "name": "generate_chart",
        "description": "Generate a chart from the shark attacks data",
        "parameters": {
            "type": "object",
            "properties": {
                "chart_type": {"type": "string", "enum": ["bar", "line", "pie", "hist"]},
                "x": {"type": "string"},
                "y": {"type": "string"},
                "bins": {"type": "integer"}
            },
            "required": ["chart_type", "x"]
        }
    }
}]

# ── AI-Endpoint ────────────────────────────────────────────────────────────────


@app.route('/chat', methods=['POST'])
def chat():
    user_msg = request.json.get('message')
    messages = [
        {"role": "system", "content": "You are an assistant for analyzing global shark attack data."},
        {"role": "user", "content": user_msg}
    ]
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )
    msg = resp.choices[0].message
    if msg.tool_calls:
        tool_call = msg.tool_calls[0]
        fn_name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)
        fn_resp = globals()[fn_name](**args)

        messages.append(msg.model_dump())
        messages.append({
            "role": "tool",
            "name": fn_name,
            "content": json.dumps(fn_resp),
            "tool_call_id": tool_call.id
        })
        follow = client.chat.completions.create(
            model="gpt-4o-mini", messages=messages)
        return jsonify({"reply": follow.choices[0].message.content, **fn_resp})
    return jsonify({"reply": msg.content})

# ── Dashboard-Route ──────────────────────────────────────────────────────────


@app.route('/')
def index():
    # PNGs neu erzeugen
    plot_attacks_per_year(df)
    plot_fatal_vs_nonfatal(df)
    plot_top_countries(df)
    plot_age_distribution(df)
    graphs = [
        ('attacks_per_year.png', 'Attacks per Year'),
        ('fatal_vs_nonfatal.png', 'Fatal vs. Non-fatal'),
        ('top_countries.png', 'Top Countries'),
        ('age_distribution.png', 'Age Distribution')
    ]
    records = df.to_dict(orient='records')
    columns = df.columns.tolist()
    map_url = url_for('static', filename='map.html')
    return render_template('index.html',
                           graphs=graphs,
                           columns=columns,
                           records=records,
                           map_url=map_url)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
