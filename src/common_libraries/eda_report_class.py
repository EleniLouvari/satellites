from import_libraries import *
import global_variables as gb_l
import libraries.common_libraries.generic_library as cm_l
import libraries.common_libraries.geom_library as geom_l

class EdaReport:
    def __init__(self, input_data, report_folder=None, report_name=None):
        self.input_data = input_data
        self.input_type = None
        self.gis_file = None
        self.file_name = None
        self.gdf = None
        self.save_folder = None
        self.report_folder = report_folder
        self.report_name = report_name
        self.layers = [None]
        self.layer_info = {}
        self.relative_paths = {}
        self.object_cols = None
        self.numerical_cols = None
        self.info_df_geom = None
        self.info_df_obj = None
        self.info_df_num = None
        self.palette = "tab10"
        self.color_maps = {}
        self.valid_extensions = ['.shp', '.gpkg', '.gdb']
        self.errors = []
        self.html_report_file = None

        # Create paths
        self.setup_paths()
        # Initialize logging
        self.setup_logging()
        # Suppress all warnings
        warnings.filterwarnings('ignore')

        self.set_plot_style()


    def setup_paths(self):
        try:
            cm_l.print_formatted_txt(f"Creating paths...", "SECTION")

            if isinstance(self.input_data, gpd.GeoDataFrame):
                if self.report_folder is None:
                    raise ValueError("'report_folder' must be defined")
                if self.report_name is None:
                    raise ValueError("'report_name' must be defined")

                self.input_type = "geodataframe"
                self.file_name = self.report_name
                self.gdf = self.input_data.copy()
                self.layers = [self.report_name]
                self.report_folder = os.path.join(self.report_folder, f"report_{self.report_name}")

            elif isinstance(self.input_data, str):
                if cm_l.is_valid_path_filename(self.input_data):
                    self.input_type = "file"
                    self.file_name = os.path.splitext(os.path.basename(self.input_data))[0]
                    self.report_folder = os.path.join(os.path.dirname(self.input_data), f"report_{self.file_name}")
                else:
                    raise FileNotFoundError(f"File not found: {self.input_data}")

        except Exception as e:
            msg = f"Error setting paths: {str(e)}"
            self.errors.append(msg)


    def setup_logging(self):
        """Set up logging configuration"""
        try:
            cm_l.create_folder(self.report_folder)

            # Create timestamp for log file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(self.report_folder, f"eda_report_{timestamp}.log")

            # Configure logging
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(levelname)s - %(message)s",
                handlers=[
                    logging.FileHandler(log_file),
                    logging.StreamHandler()
                ]
            )
            self.logger = logging.getLogger(__name__)
            self.logger.info("Initializing EDA Report generation")
        except Exception as e:
            print(f"Error setting log parameters: {str(e)}")
            raise


    def release_logger(self, logger):
        """Release all handlers associated with the logger."""
        handlers = logger.handlers[:]
        for handler in handlers:
            handler.flush()
            handler.close()
            logger.removeHandler(handler)


    def set_plot_style(self):
        try:
            plt.rcParams['font.family'] = "sans-serif"
            plt.rcParams['font.sans-serif'] = ['Verdana', 'DejaVu Sans', 'Arial', 'Helvetica']
            plt.rcParams['font.size'] = 10
            plt.rcParams['axes.titlesize'] = 12
            plt.rcParams['axes.labelsize'] = 11

            font_paths = font_manager.findSystemFonts(fontpaths=None, fontext="ttf")
            verdana_available = any("verdana" in font.lower() for font in font_paths)
            if not verdana_available:
                self.logger.warning("Verdana font not found. Using system default sans-serif font.")
        except Exception as e:
            msg = f"Error setting plot style: {str(e)}"
            self.errors.append(msg)
            self.logger.error(msg)


    def read_data_and_create_report(self):
        try:
            self.logger.info("Starting data reading and report creation process")

            if self.input_type == "geodataframe":
                for layer in self.layers:
                    self.process_layer(input_type=self.input_type, layer_name=layer)
            else:
                _, file_extension = os.path.splitext(self.input_data)

                if file_extension == ".zip":
                    self.logger.info(f"Processing zip file: {self.input_data}")
                    self.gis_file = cm_l.unzip_gis_file(self.input_data, self.valid_extensions)
                elif file_extension in self.valid_extensions:
                    self.gis_file = self.input_data
                else:
                    raise ValueError(f"Invalid file format. Only {self.valid_extensions} are allowed!")

                _, file_extension = os.path.splitext(self.gis_file)
                if file_extension in ['.gpkg', '.gdb']:
                    self.layers = cm_l.get_gdb_layers(self.gis_file)
                    self.logger.info(f"Found {len(self.layers)} layers in {self.gis_file}")
                else:
                    self.layers = [None]

                for layer in self.layers:
                    self.process_layer(input_type=self.input_type, layer_name=layer)

        except Exception as e:
            msg = f"Error in read_data_and_create_report: {str(e)}"
            self.errors.append(msg)
            self.logger.error(msg)
            raise


    def process_layer(self, input_type, layer_name):
        cm_l.print_formatted_txt(f"Process layer {layer_name}...", "SECTION")
        try:
            if input_type == "file":
                if layer_name:
                    self.logger.info(f"Processing layer: {layer_name} from file: {self.gis_file}")
                    gdf = cm_l.read_data(self.gis_file, layer=layer_name)
                    self.save_folder = os.path.join(self.report_folder, layer_name)
                    self.report_name = f"report_{layer_name}"
                    relative_path = os.path.join(layer_name, self.report_name)
                else:
                    self.logger.info(f"Processing file: {self.gis_file}")
                    gdf = cm_l.read_data(self.gis_file)
                    self.save_folder = self.report_folder
                    self.report_name = f"report_{self.file_name}"
                    relative_path = self.report_name
                layer_name = layer_name or self.file_name
            else:
                self.logger.info("Processing geodataframe")
                layer_name = layer_name or self.report_name or "geodataframe"
                self.report_name = f"report_{layer_name}" if not str(layer_name).startswith("report_") else str(layer_name)
                self.save_folder = os.path.join(self.report_folder, str(layer_name))
                gdf = self.gdf.copy()
                relative_path = os.path.join(os.path.relpath(self.save_folder, self.report_folder), self.report_name)

            self.layer_info[layer_name] = {
                'save_folder': self.save_folder,
                'report_name': self.report_name,
                'relative_path': relative_path,
                'num_records': len(gdf.index)
            }

            if "geometry" in gdf:
                cm_l.delete_folder(self.save_folder)
                cm_l.create_folder(self.save_folder)
                self.logger.info(f"Created folder: {self.save_folder}")

                gdf = cm_l.fix_json_chars_in_column_names(gdf)
                gdf = gdf.rename(columns=lambda x: x.lower())

                self.gdf, self.object_cols, self.numerical_cols = self.preprocess(gdf)
                self.logger.info(f"Preprocessed data: {len(self.object_cols)} object columns, {len(self.numerical_cols)} numerical columns")

                self.create_thumbnail_map(layer_name)
                self.create_plots()
                self.generate_html_report()
            else:
                self.logger.error("Geometry column is missing in the dataset")
                raise ValueError("Geometry column is missing")

        except Exception as e:
            msg = f"Error processing layer {layer_name}: {str(e)}"
            self.errors.append(msg)
            self.logger.error(msg)
            raise


    def preprocess(self, gdf):
        cm_l.print_formatted_txt("Checking data process...", "SUBSECTION")

        geoms = gdf['geometry']
        crs = gdf.crs
        gdf.drop(columns=['geometry'], inplace=True)
        gdf, object_cols, numerical_cols = cm_l.get_column_types(gdf, raise_error=False)
        gdf = cm_l.fill_nulls(gdf, object_cols, numerical_cols, gb_l.string_null_values_list, [-np.inf, np.inf], "NONE", np.nan)
        gdf['geometry'] = geoms
        gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=crs)

        cols_to_drop = []
        for col in numerical_cols:
            if gdf[col].isnull().all():
                msg = f"Column '{col}' is entirely null. It will be excluded!"
                print(msg)
                self.logger.info(msg)
                cols_to_drop.append(col)

        for col in object_cols:
            if gdf[col].str.lower().eq("none").all() or gdf[col].isnull().all():
                msg = f"Column '{col}' is entirely null. It will be excluded!"
                print(msg)
                self.logger.info(msg)
                cols_to_drop.append(col)

        object_cols = [x for x in object_cols if x not in cols_to_drop]
        numerical_cols = [x for x in numerical_cols if x not in cols_to_drop]
        gdf = gdf.drop(columns=cols_to_drop)

        return gdf, object_cols, numerical_cols


    def create_color_map(self, column, top_n=10):
        value_counts = self.gdf[column].value_counts()
        top_categories = value_counts.nlargest(top_n).index

        unique_colors = plt.get_cmap(self.palette)(range(top_n))
        color_dict = {category: unique_colors[i] for i, category in enumerate(top_categories)}
        color_dict['Others'] = "lightgrey"

        self.color_maps[column] = color_dict


    def create_plots(self):
        try:
            self.logger.info("Creating plots")
            gdf_for_geom_info = self.gdf.copy()
            if gdf_for_geom_info.crs is not None and not gdf_for_geom_info.crs.is_projected:
                self.logger.info("Projecting geometry diagnostics to EPSG:3857 because source CRS is geographic")
                gdf_for_geom_info = gdf_for_geom_info.to_crs(epsg=3857)
            self.info_df_geom = cm_l.get_geometries_info(gdf_for_geom_info, figsize=(15, 8), plot_graphs=False, verbose=False, save_folder=self.save_folder)
            geometry_plot_path = f"{self.save_folder}/geometry_barplot.png"
            coords_plot_path = f"{self.save_folder}/coords_barplot.png"
            if os.path.exists(coords_plot_path) and not os.path.exists(geometry_plot_path):
                geometry_plot_path = coords_plot_path

            self.info_df_obj, self.info_df_num = cm_l.df_info_per_column(self.gdf, self.object_cols, self.numerical_cols, verbose=False)

            for col in self.info_df_obj['Column']:
                self.create_color_map(col)

            self.info_df_obj.drop(columns=['Data type'], inplace=True)
            for col in list(self.info_df_obj['Column']):
                self.logger.info(f"Creating plot for object column: {col}")
                cm_l.plot_stat_object(self.gdf, col, show_common=10, palette=self.color_maps[col], figsize=(15, 6), save_folder=self.save_folder)

            for col in list(self.info_df_num['Column']):
                self.logger.info(f"Creating plot for numeric column: {col}")
                cm_l.plot_stat_numeric(self.gdf, col, exclude_zeros=False, figsize=(15, 8), save_folder=self.save_folder)
        except Exception as e:
            msg = f"Error creating plots: {str(e)}"
            self.errors.append(msg)
            self.logger.error(msg)
            raise


    def create_thumbnail_map(self, layer_name):
        fig, ax = plt.subplots(figsize=(6, 6))
        self.gdf.plot(ax=ax)
        if layer_name != "geodataframe":
            ax.set_title(f"{layer_name}")
        ax.axis("off")
        thumbnail_filename = f"{layer_name}_thumbnail.png"
        thumbnail_path = os.path.join(self.save_folder, thumbnail_filename)
        plt.savefig(thumbnail_path, bbox_inches="tight", dpi=100)
        plt.close()

        self.relative_paths[layer_name] = {
            'thumbnail': os.path.join(os.path.relpath(self.save_folder, self.report_folder), thumbnail_filename)
        }


    def create_map_plot(self, column, col_type):
        fig, ax = plt.subplots(1, 1, figsize=(20, 20))

        if col_type == "object":
            color_dict = self.color_maps.get(column)

            self.gdf['color_category'] = self.gdf[column].apply(
                lambda x: color_dict[x] if x in color_dict else "lightgrey"
            )

            self.gdf.plot(color=self.gdf['color_category'], ax=ax, legend=True)

            from matplotlib.lines import Line2D
            legend_elements = [Line2D([0], [0], marker="o", color="w", label=cat,
                                    markerfacecolor=color, markersize=10)
                            for cat, color in color_dict.items()]
            ax.legend(handles=legend_elements, bbox_to_anchor=(1, 1))

        elif col_type == "numeric":
            norm = mcolors.Normalize(vmin=self.gdf[column].min(), vmax=self.gdf[column].max())
            self.gdf.plot(column=column, cmap="coolwarm", scheme="quantiles", legend=True, ax=ax)

        else:
            self.gdf.plot(ax=ax)

        ax.set_title(f"Map plot for column: {column}")
        map_img_path = f"{self.save_folder}/{column}_map.png"
        plt.savefig(map_img_path, bbox_inches="tight")
        plt.close()

        return map_img_path


    def create_html_template(self):
        html_template = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Data Analysis Report: {title}</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f0f2f5;
                    color: #333;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                    padding: 20px;
                    border-radius: 8px;
                }}
                h1, h2 {{
                    color: #2c3e50;
                    border-bottom: 2px solid #3498db;
                    padding-bottom: 10px;
                }}
                .tabs {{
                    display: flex;
                    border-bottom: 2px solid #3498db;
                    margin-bottom: 20px;
                }}
                .tab {{
                    padding: 10px 20px;
                    cursor: pointer;
                    background-color: #ecf0f1;
                    border-top-left-radius: 8px;
                    border-top-right-radius: 8px;
                    margin-right: 10px;
                    transition: background-color 0.3s;
                }}
                .tab.active {{
                    background-color: #3498db;
                    color: white;
                    font-weight: bold;
                }}
                .tab-content {{
                    display: none;
                }}
                .tab-content.active {{
                    display: block;
                }}
                .column-section {{
                    display: flex;
                    flex-wrap: wrap;
                    margin-bottom: 30px;
                    border: 1px solid #ddd;
                    padding: 20px;
                    align-items: flex-start;
                    background-color: #fff;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .info-table {{
                    flex: 1;
                    min-width: 300px;
                    margin-right: 20px;
                    align-self: flex-start;
                }}
                .plot-image {{
                    flex: 2;
                    min-width: 300px;
                    display: flex;
                    justify-content: flex-start;
                    align-items: flex-start;
                }}
                .plot-image img {{
                    max-width: 100%;
                    height: auto;
                    object-fit: contain;
                    border-radius: 4px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                table {{
                    border-collapse: separate;
                    border-spacing: 0;
                    width: 100%;
                    margin-top: 10px;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                    overflow: hidden;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 12px;
                    text-align: left;
                }}
                th {{
                    background-color: #3498db;
                    color: white;
                    font-weight: bold;
                }}
                tr:nth-child(even) {{
                    background-color: #f2f2f2;
                }}
                .summary-stats {{
                    margin-top: 20px;
                    padding: 15px;
                    background-color: #ecf0f1;
                    border-radius: 4px;
                }}
                .toggle-btn {{
                    background-color: #3498db;
                    color: white;
                    border: none;
                    padding: 10px 15px;
                    border-radius: 4px;
                    cursor: pointer;
                    margin-top: 10px;
                }}
                .toggle-btn:hover {{
                    background-color: #2980b9;
                }}
                .hidden {{
                    display: none;
                }}
                .visible {{
                    display: block;
                }}
            </style>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        </head>
        <body>
            <div class="container">
                <h1>Data Analysis Report for: {title}</h1>
                {summary_stats}

                <!-- Tabs Section -->
                <div class="tabs">
                    <div class="tab active" onclick="showTab('geometry-columns')">Geometry Column</div>
                    <div class="tab" onclick="showTab('object-columns')">Object Columns</div>
                    <div class="tab" onclick="showTab('numeric-columns')">Numeric Columns</div>
                </div>

                <!-- Tab Contents -->
                <div id="geometry-columns" class="tab-content active">
                    {geometry_column_sections}
                </div>
                <div id="object-columns" class="tab-content">
                    {object_column_sections}
                </div>
                <div id="numeric-columns" class="tab-content">
                    {numeric_column_sections}
                </div>
            </div>

            <script>
                function showTab(tabId) {{
                    // Hide all tab contents
                    var tabContents = document.getElementsByClassName('tab-content');
                    for (var i = 0; i < tabContents.length; i++) {{
                        tabContents[i].classList.remove('active');
                    }}

                    // Remove active class from all tabs
                    var tabs = document.getElementsByClassName('tab');
                    for (var i = 0; i < tabs.length; i++) {{
                        tabs[i].classList.remove('active');
                    }}

                    // Show selected tab content
                    document.getElementById(tabId).classList.add('active');

                    // Set active class on clicked tab
                    event.currentTarget.classList.add('active');
                }}

                // Function to show/hide the stats div for a specific column
                function toggleStats(statsId) {{
                    var statsDiv = document.getElementById(statsId);
                    var button = document.getElementById(statsId.replace('div', 'btn'));
                    if (statsDiv.classList.contains('hidden')) {{
                        statsDiv.classList.remove('hidden');
                        button.textContent = 'Hide Additional Stats';
                    }} else {{
                        statsDiv.classList.add('hidden');
                        button.textContent = 'Show Additional Stats';
                    }}
                }}

                // Function to show/hide the map div for a specific column
                function toggleMap(mapId) {{
                    var mapDiv = document.getElementById(mapId);
                    var button = document.getElementById(mapId.replace('div', 'btn'));
                    if (mapDiv.classList.contains('hidden')) {{
                        mapDiv.classList.remove('hidden');
                        button.textContent = 'Hide Map';
                    }} else {{
                        mapDiv.classList.add('hidden');
                        button.textContent = 'Show Map';
                    }}
                }}
            </script>
        </body>
        </html>
        """
        return html_template


    def create_column_section(self, column, info_df, col_type):
        # Create unique IDs for each column based on the column name
        column_safe = column.replace(" ", "_").lower()  # Create a safe ID name

        section = f"<h2>Column: '{column}'</h2>"
        section += '<div class="column-section">'

        # Info table
        section += '<div class="info-table">' # start column-section
        section += '<table>'
        for index, row in info_df[info_df['Column'] == column].iterrows():
            for key, value in row.items():
                if key != "Column":
                    section += f'<tr><th>{key}</th><td>{value}</td></tr>'
        section += '</table>'

        # Add toggle button for additional stats and map, with unique IDs
        section += f'<button id="btnStats_{column_safe}" class="toggle-btn" onclick="toggleStats(\'divStats_{column_safe}\')">Show Additional Stats</button>'
        section += f'<button id="btnMap_{column_safe}" class="toggle-btn" onclick="toggleMap(\'divMap_{column_safe}\')">Show Map</button>'

        # Additional Stats Section (initially hidden)
        section += f'<div id="divStats_{column_safe}" class="additional-stats hidden">'
        if col_type == "numeric":
            desc_stats = self.gdf[column].describe().to_dict()
            section += '<h3>Descriptive Statistics</h3>'
            section += '<table>'
            for stat, value in desc_stats.items():
                formatted_value = f'{value:.2f}' if isinstance(value, float) else str(value)
                section += f'<tr><th>{stat}</th><td>{formatted_value}</td></tr>'
            section += '</table>'
        else:
            value_counts = self.gdf[column].value_counts().head(10).to_dict()
            section += '<h3>Top 10 Values</h3>'
            section += '<table>'
            for value, count in value_counts.items():
                section += f'<tr><th>{value}</th><td>{count}</td></tr>'
            section += '</table>'
        section += '</div>'
        section += '</div>'  # Info table

        # Plot image
        section += '<div class="plot-image">'
        img_path = f"{self.save_folder}/{column}_barplot.png"
        if column == "geometry" and not os.path.exists(img_path):
            img_path = f"{self.save_folder}/coords_barplot.png"
        with open(img_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        section += f'<img src="data:image/png;base64,{encoded_string}" alt="{column} plot">'
        section += '</div>'

        # Map Section (initially hidden)
        section += f'<div id="divMap_{column_safe}" class="map-section hidden">'
        map_img_path = self.create_map_plot(column, col_type)
        with open(map_img_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        section += f'<img src="data:image/png;base64,{encoded_string}" alt="{column} map" style="max-width: 100%;">'

        if column == "geometry":
            gdf_az = self.gdf.copy()
            gdf_az['geom_type'] = gdf_az['geometry'].geom_type
            gdf_az = gdf_az[gdf_az['geom_type'].str.lower().str.contains("line")].copy()
            if not gdf_az.empty:
                _ = geom_l.plot_azimuth_rose(gdf_az, azimuth_column="line_azimuth", step=15, title="Line Azimuth Distribution (15° intervals)", fig_size=(12, 12), save_folder=self.save_folder)
                az_img_path = f"{self.save_folder}/rose_diagram.png"
                with open(az_img_path, "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode()
                section += f'<img src="data:image/png;base64,{encoded_string}" alt="{column} line azimuths" style="max-width: 100%;">'

        section += '</div>'  # map

        section += '</div>'  # end column-section

        return section


    def generate_html_report(self):
        cm_l.print_formatted_txt("Creating html process...", "SUBSECTION")
        html_template = self.create_html_template()

        total_records = len(self.gdf)
        total_columns = len(self.gdf.columns)
        object_columns = len(self.info_df_obj)
        numeric_columns = len(self.info_df_num)

        summary_stats = f"""
        <div class="summary-stats">
            <h2>Dataset Summary</h2>
            <p>Total number of records: {total_records}</p>
            <p>Number of columns: {total_columns} ({object_columns} object, {numeric_columns} numeric)</p>
        </div>
        """

        # Generate column sections separately for object and numeric columns
        geometry_column_sections = self.create_column_section("geometry", self.info_df_geom, col_type="geometry")

        object_column_sections = ""
        for column in self.info_df_obj['Column']:
            object_column_sections += self.create_column_section(column, self.info_df_obj, col_type="object")

        numeric_column_sections = ""
        for column in self.info_df_num['Column']:
            numeric_column_sections += self.create_column_section(column, self.info_df_num, col_type="numeric")

        # Combine all parts
        html_content = html_template.format(
            title=f"{self.report_name.replace('report_', '')}",
            summary_stats=summary_stats,
            geometry_column_sections=geometry_column_sections,
            object_column_sections=object_column_sections,
            numeric_column_sections=numeric_column_sections
        )

        # Save the HTML content to a file
        output_path = f"{self.save_folder}/{self.report_name}.html"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        self.logger.info(f"Data layer report generated successfully '{self.report_name}'!")


    def create_overview_report(self):
        cm_l.print_formatted_txt("Creating overview report...", "SECTION")

        # HTML template for the overview report
        html_template = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Overview Report: {title}</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif, Arial;
                    margin: 0;
                    padding: 20px;
                    background-color: #f0f2f5;
                    color: #333;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                    padding: 20px;
                    border-radius: 8px;
                }}
                h1, h2 {{
                    color: #2c3e50;
                    border-bottom: 2px solid #3498db;
                    padding-bottom: 10px;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-top: 20px;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 12px;
                    text-align: left;
                }}
                th {{
                    background-color: #3498db;
                    color: white;
                }}
                tr:nth-child(even) {{
                    background-color: #f2f2f2;
                }}
                .thumbnail {{
                    max-width: 200px;
                    max-height: 200px;
                    object-fit: contain;
                }}
                .layer-info {{
                    display: flex;
                    align-items: center;
                    margin-bottom: 20px;
                }}
                .layer-info img {{
                    margin-right: 20px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Overview Report: {title}</h1>
                <h2>General Information</h2>
                <p>Total number of layers: {total_layers}</p>
                {layers_info}
                <h2>Layer Reports</h2>
                <table>
                    <tr>
                        <th>Layer</th>
                        <th>Thumbnail</th>
                        <th>Report Link</th>
                        <th>Total Records</th>
                    </tr>
                    {layer_rows}
                </table>
            </div>
        </body>
        </html>
        """

        # Generate information for each layer
        layers_info = ""
        total_layers = len(self.layer_info) if self.layer_info else 0
        layer_rows = ""
        for layer_name, layer_data in self.layer_info.items():
            relative_report_path = layer_data['relative_path']
            relative_thumbnail_path = self.relative_paths[layer_name]['thumbnail']
            num_records = layer_data['num_records']

            layer_rows += f"""
            <tr>
                <td>{layer_name}</td>
                <td><img src="{relative_thumbnail_path}" alt="{layer_name} thumbnail" class="thumbnail"></td>
                <td><a href="{relative_report_path}.html">View Detailed Report</a></td>
                <td>{num_records}</td>
            </tr>
            """

        # Combine all parts
        html_content = html_template.format(
            title=self.file_name,
            total_layers=total_layers,
            layers_info=layers_info,
            layer_rows=layer_rows
        )

        # Save the HTML content to a file
        output_path = os.path.join(self.report_folder, "overview_report.html")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        self.html_report_file = output_path
        self.logger.info("Overview report generated succesfully!")


    def create_eda_report(self):
        try:
            self.logger.info("Starting EDA report creation")
            self.read_data_and_create_report()
            self.create_overview_report()

            self.logger.info("EDA report creation completed successfully")
            # Ensure the logger releases the log file.
            self.release_logger(self.logger)

            if self.gis_file:
                # After you're done using the GeoDataFrame, delete it
                del self.gdf
                # Force garbage collection to release resources
                gc.collect()
                # cm_l.delete_file(self.gis_file)

            return self.html_report_file
        except Exception as e:
            msg = f"Error creating EDA report: {str(e)}"
            self.errors.append(msg)
            self.logger.error(msg)
            return self.errors
