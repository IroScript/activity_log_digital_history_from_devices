const { GLib, Gio, Shell } = imports.gi;

export default class UnsafeModeExtension {
    constructor() {
        this._timeoutId = null;
    }

    enable() {
        global.context.unsafe_mode = true;
        
        this._timeoutId = GLib.timeout_add_seconds(GLib.PRIORITY_DEFAULT, 1, () => {
            try {
                let focusWindow = global.display.get_focus_window();
                let appName = "Linux App";
                let windowTitle = "Linux Window";
                
                if (focusWindow) {
                    windowTitle = focusWindow.get_title() || "Linux Window";
                    
                    let tracker = Shell.WindowTracker.get_default();
                    let app = tracker ? tracker.get_window_app(focusWindow) : null;
                    
                    if (app) {
                        appName = app.get_name();
                    } else {
                        appName = focusWindow.get_wm_class() || "Linux App";
                    }
                }
                
                let data = JSON.stringify({ app: appName, title: windowTitle });
                let file = Gio.File.new_for_path('/tmp/openrecall_active_window.json');
                file.replace_contents_bytes_async(
                    new GLib.Bytes(data),
                    null,
                    false,
                    Gio.FileCreateFlags.REPLACE_DESTINATION,
                    null,
                    null
                );
            } catch (e) {
                console.error("OpenRecall Extension Error: " + e);
            }
            return GLib.SOURCE_CONTINUE;
        });
    }

    disable() {
        global.context.unsafe_mode = false;
        if (this._timeoutId) {
            GLib.source_remove(this._timeoutId);
            this._timeoutId = null;
        }
    }
}
