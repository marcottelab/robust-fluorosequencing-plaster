function z() {
    // Inspired by https://github.com/hyperhype/hyperscript
    let a = arguments[0];
    let start = 1;
    if (typeof a === 'string') {
    }
    else if(!a) {
        // If the first argument is a boolean-like then evaluate it
        // to see if the whole block should be skipped.
        return;
    }
    else {
        a = arguments[start];
        start++;
    }
    let m = a.split(/(?=[\.#])/g);
    let e = document.createElement(m[0]);
    m.slice(1).forEach(function (i) {
        let term = i.substring(1);
        if(i[0] === '.') {
            e.className += ' ' + term;
        }
        else if(i[0] === '#') {
            e.setAttribute('id', term);
        }
    });
    for(let i=start; i<arguments.length; i++) {
        let a = arguments[i];
        if(typeof a === 'string') {
            e.appendChild(document.createTextNode(a));
        }
        else if(a instanceof Array) {
            function unwind(arr) {
                for(let j=0; j<arr.length; j++) {
                    if(typeof arr[j] !== 'undefined') {
                        if(arr[j] instanceof Array) {
                            unwind(arr[j]);
                        }
                        else {
                            e.appendChild(arr[j]);
                        }
                    }
                }
            }
            unwind(a);
        }
        else if(a instanceof HTMLElement) {
            e.appendChild(a);
        }
        else if(a instanceof Object) {
            for(let key in a) {
                if(a.hasOwnProperty(key)) {
                    if(key == 'data') {
                        for(let d in a[key]) {
                            e.dataset[d] = a[key][d];
                        }
                    }
                    else {
                        e.setAttribute(key, a[key]);
                    }
                }
            }
        }
        else {
            if(typeof a !== 'undefined') {
                e.appendChild(document.createTextNode(a.toString()));
            }
        }
    }
    return e;
}

function timeSince(date) {
    let seconds = Math.floor((new Date()) / 1000 - date);
    let intervals = [
        [31536000, "year"],
        [2592000, "month"],
        [86400, "day"],
        [3600, "hour"],
        [60, "minute"],
        [1, "second"],
    ];
    for(let _i in intervals) {
        let i = intervals[_i];
        let interval = Math.floor(seconds / i[0]);
        if(interval >= 1 || i[0] == 1) {
            return interval + " " + i[1] + (interval > 1 ? "s" : "");
        }
    }
}
