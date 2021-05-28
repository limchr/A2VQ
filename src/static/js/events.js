c2.addEventListener('mousedown', function(e) {
    if(selection_begin) return;
    selection_begin = true;
    selection_start = get_mouse_xy(c2, e);
    console.log('selection was started on '+selection_start.x+' '+selection_start.y);

  }, true);

c2.addEventListener('mousemove', function(e) {
    if(selection_begin) {
      selection_moved = get_mouse_xy(c2, e);
      console.log('selection was moved on '+selection_moved.x+' '+selection_moved.y);

      draw_selections();
      min_x = Math.min(selection_start.x, selection_moved.x)
      max_x = Math.max(selection_start.x, selection_moved.x)
      min_y = Math.min(selection_start.y, selection_moved.y)
      max_y = Math.max(selection_start.y, selection_moved.y)

      draw_selection(min_x,min_y,(max_x-min_x),(max_y-min_y));

    }


  }, true);


c2.addEventListener('mouseup', function(e) {
    if(selection_begin) {
      selection_begin = false;
      var mouse = get_mouse_xy(c2, e);

      if (mouse.x == selection_start.x && mouse.y == selection_start.y){
        console.log('selection too small, skipping');
        return;
      }

      console.log('selection was ended on '+mouse.x+' '+mouse.y);

      min_x = Math.min(selection_start.x, mouse.x)
      max_x = Math.max(selection_start.x, mouse.x)
      min_y = Math.min(selection_start.y, mouse.y)
      max_y = Math.max(selection_start.y, mouse.y)

      selections.push([min_x,max_x,min_y,max_y]);

      draw_selections();


    }

  }, true);
