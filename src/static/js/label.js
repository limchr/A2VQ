document.addEventListener("DOMContentLoaded", function(event) {
  w = $('#canvas_container').width();
  dim_canvas = [w,w*0.75];
  dim_img = [dim_canvas[0]/15,dim_canvas[0]/15];
  c.width = dim_canvas[0];
  c.height = dim_canvas[1];

  ctx.clearRect(0, 0, c.width, c.height);
  ctx.fillStyle = 'rgba(255, 255, 255, 1)';
  ctx.fillRect(0, 0, c.width, c.height);


  c2.width = dim_canvas[0];
  c2.height = dim_canvas[1];
  ctx2.clearRect(0, 0, c2.width, c2.height);


  draw_imgs();

  //register callbacks
  $('#label_button').click(function(){label_selected();selections = []; draw_everything();});
  $('#remove_button').click(function(){selections = []; draw_everything();});
    $('#new_class_button').click(function(){window.location.replace('/add_class_'+$('#class_label').val());});
});




function label_selected(){

  class_name = $('#label_name').val();

  selected = []
  for (var i=0;i<Object.keys(samples).length;++i) {
    for (var s=0;s<selections.length;++s){
      if(selection_intersects(selections[s],[samples[i].embedding[0]*(dim_canvas[0]-dim_img[0]) + dim_img[0]/2  , samples[i].embedding[1]*(dim_canvas[1]-dim_img[1]) + dim_img[1]/2 ])) {
        if($.inArray(samples[i].id, selected) === -1) {
          selected.push(samples[i].id);
        }
      }
    }
  }
  if(selected.length == 0){
  return;
  }

  $('#label_button').addClass('disabled');
  $('#next_view_button').addClass('disabled');

  $('#label_button').addClass('btn-warning');
  $('#next_view_button').addClass('btn-warning');
  $('#next_view_link').attr('href', '/');

   $.ajax({
          type: "POST",
          cache: false,
          data:{'label': class_name, 'ids': selected, 'num_rects': selections.length},
          url: '/add_labels',
          dataType: "json",
          success: function(data) {
            if(data["success"] == 'false'){
              window.open(data["open_page"],"_self");
            }
              console.log(data);
                $('#label_button').removeClass('disabled');
                $('#next_view_button').removeClass('disabled');
                $('#label_button').removeClass('btn-warning');
                $('#next_view_button').removeClass('btn-warning');
                $('#label_button').addClass('btn-primary');
                $('#next_view_button').addClass('btn-success');
                $('#next_view_link').attr('href', 'a2vq');
          },
          error: function(jqXHR) {
              alert("error: " + jqXHR.status);
              console.log(jqXHR);
                $('#label_button').removeClass('disabled');
                $('#next_view_button').removeClass('disabled');
                $('#label_button').removeClass('btn-warning');
                $('#next_view_button').removeClass('btn-warning');
                $('#label_button').addClass('btn-primary');
                $('#next_view_button').addClass('btn-success');
                $('#next_view_link').attr('href', '/');
          }
      });

        for (var i = 0; i<selected.length; ++i){
            samples[selected[i]].label = class_name;
        }

}
