import omni.ext
import omni.ui as ui

#--------------------------#
#   user event functions   #
#--------------------------#


def segment_fn():
    print("[segment_fn] procedure started")

def cleanup_fn():
    print("[cleanu_fn] procedure started")


#--------------------------#
#   main extension class   #
#--------------------------#

class MyExtension(omni.ext.IExt):

    class_labels : list = []

    def on_startup(self, ext_id):
        print("[omni.vseg] VSeg on_startup")

        self._window = ui.Window("VSeg", width=450, height=700)
        with self._window.frame:
            with ui.VStack():

                self.button_clear_labels = ui.Button("Clear Labels")

                with ui.HStack():
                    self.field_class_label = ui.StringField()
                    self.button_add_new_label = ui.Button("+")

                self.button_segment = ui.Button("Compute Segments")
                self.button_cleanup = ui.Button("Clear Segments")
                self.button_toggle_prims = ui.Button("Show/Hide Voxels")
                    

    def on_shutdown(self):
        print("[omni.vseg] VSeg on_shutdown")
