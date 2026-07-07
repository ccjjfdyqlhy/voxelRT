#ifndef WINDOW_H
#define WINDOW_H

#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/Xatom.h>
#include <X11/keysym.h>
#include <X11/extensions/XShm.h>
#include <sys/shm.h>
#include <sys/ipc.h>
#include <cstring>
#include <cstdio>
#include <unordered_set>

class WindowX11 {
public:
    using EventCallback = bool (*)(void*);

    WindowX11(int w, int h, const char* title = "Voxel Ray Tracer")
        : width_(w), height_(h), running_(true)
    {
        display_ = XOpenDisplay(nullptr);
        if (!display_) {
            fprintf(stderr, "Cannot open display\n");
            running_ = false;
            return;
        }

        screen_ = DefaultScreen(display_);
        root_ = RootWindow(display_, screen_);

        // 创建窗口
        win_ = XCreateSimpleWindow(display_, root_, 0, 0, w, h, 0,
                                    WhitePixel(display_, screen_),
                                    BlackPixel(display_, screen_));

        // 设置 WM_DELETE_WINDOW 协议
        Atom wmDelete = XInternAtom(display_, "WM_DELETE_WINDOW", False);
        XSetWMProtocols(display_, win_, &wmDelete, 1);

        // 关注输入事件
        XSelectInput(display_, win_,
                     ExposureMask | KeyPressMask | KeyReleaseMask |
                     ButtonPressMask | ButtonReleaseMask | PointerMotionMask |
                     StructureNotifyMask);

        // 显示窗口
        XMapWindow(display_, win_);
        XStoreName(display_, win_, title);

        // 创建 GC
        gc_ = XCreateGC(display_, win_, 0, nullptr);

        // 加载字体
        font_ = XLoadQueryFont(display_, "fixed");
        if (!font_) font_ = XLoadQueryFont(display_, "9x15");
        if (!font_) font_ = XLoadQueryFont(display_, "6x13");
        if (font_) XSetFont(display_, gc_, font_->fid);

        // 创建共享内存图像
        createShmImage(w, h);

        // 初始化键盘状态
        keys_.clear();

        // 鼠标状态
        mouseDX_ = mouseDY_ = 0;
        mouseGrabbed_ = false;
    }

    ~WindowX11() {
        if (shminfo_.shmaddr) {
            XShmDetach(display_, &shminfo_);
            shmdt(shminfo_.shmaddr);
            shmctl(shminfo_.shmid, IPC_RMID, nullptr);
        }
        if (img_) XDestroyImage(img_);
        if (font_) XFreeFont(display_, font_);
        if (gc_) XFreeGC(display_, gc_);
        if (win_) XDestroyWindow(display_, win_);
        if (display_) XCloseDisplay(display_);
    }

    void processEvents() {
        mouseDX_ = 0;
        mouseDY_ = 0;
        lastKeyPressed_ = 0;

        while (XPending(display_)) {
            XEvent ev;
            XNextEvent(display_, &ev);

            if (ev.type == KeyPress) {
                lastKeyPressed_ = XLookupKeysym(&ev.xkey, 0);
            }

            if (eventCallback_ && eventCallback_(&ev))
                continue;

            switch (ev.type) {
            case Expose:
                if (ev.xexpose.count == 0) needsRedraw_ = true;
                break;

            case KeyPress: {
                KeySym ks = XLookupKeysym(&ev.xkey, 0);
                keys_.insert((unsigned int)ks);
                if (ks == XK_Shift_L || ks == XK_Shift_R) shiftPressed_ = true;
                break;
            }
            case KeyRelease: {
                // 防止按住时的自动重复
                if (XPending(display_)) {
                    XEvent nev;
                    XPeekEvent(display_, &nev);
                    if (nev.type == KeyPress &&
                        nev.xkey.keycode == ev.xkey.keycode &&
                        nev.xkey.time - ev.xkey.time < 2) {
                        break;  // 跳过自动重复
                    }
                }
                KeySym ks = XLookupKeysym(&ev.xkey, 0);
                keys_.erase((unsigned int)ks);
                if (ks == XK_Shift_L || ks == XK_Shift_R) shiftPressed_ = false;
                break;
            }
            case ButtonPress:
                if (ev.xbutton.button == Button1 && autoGrab_) {
                    grabMouse();
                }
                break;

            case MotionNotify:
                if (mouseGrabbed_) {
                    int cx = width_ / 2, cy = height_ / 2;
                    mouseDX_ = ev.xmotion.x - cx;
                    mouseDY_ = ev.xmotion.y - cy;
                    if (mouseDX_ != 0 || mouseDY_ != 0) {
                        XWarpPointer(display_, None, win_, 0, 0, 0, 0, cx, cy);
                    }
                }
                break;

            case ClientMessage:
                running_ = false;
                break;

            case ConfigureNotify:
                if (ev.xconfigure.width != width_ ||
                    ev.xconfigure.height != height_) {
                    resize(ev.xconfigure.width, ev.xconfigure.height);
                }
                break;
            }
        }
    }

    void present() {
        XShmPutImage(display_, win_, gc_, img_, 0, 0, 0, 0, width_, height_, False);
        XFlush(display_);
    }

    bool isRunning() const { return running_; }
    int width() const { return width_; }
    int height() const { return height_; }
    Display* display() const { return display_; }
    unsigned long xwindow() const { return win_; }

    bool isKeyDown(KeySym ks) const {
        return keys_.find((unsigned int)ks) != keys_.end();
    }

    bool isShiftDown() const { return shiftPressed_; }

    int mouseDX() const { return mouseDX_; }
    int mouseDY() const { return mouseDY_; }
    bool isMouseGrabbed() const { return mouseGrabbed_; }
    KeySym lastKeyPressed() const { return lastKeyPressed_; }

    void setTitle(const char* title) {
        XStoreName(display_, win_, title);
    }

    // 直接写像素到共享内存 (RGBA 格式)
    void* framebuffer() const {
        return img_ ? (void*)img_->data : nullptr;
    }

    // === Text overlay ===
    void drawText(int x, int y, const char* text, unsigned long fg = 0x00FFFFFF) {
        if (!font_) return;
        int tw = XTextWidth(font_, text, (int)strlen(text));
        int fh = font_->ascent + font_->descent;
        XSetForeground(display_, gc_, 0x00000000);
        XFillRectangle(display_, win_, gc_, x - 3, y - font_->ascent - 2,
                       tw + 6, fh + 4);
        XSetForeground(display_, gc_, fg);
        XSetFont(display_, gc_, font_->fid);
        XDrawString(display_, win_, gc_, x, y, text, (int)strlen(text));
    }

    void flush() { XFlush(display_); }

    int fontHeight() const { return font_ ? (font_->ascent + font_->descent) : 16; }

    void setEventCallback(EventCallback cb) { eventCallback_ = cb; }
    void setAutoGrab(bool enabled) { autoGrab_ = enabled; }
    void ungrabMouse() {
        if (mouseGrabbed_) {
            XUngrabPointer(display_, CurrentTime);
            mouseGrabbed_ = false;
        }
    }

    void grabMouse() {
        if (!mouseGrabbed_) {
            if (XGrabPointer(display_, win_, True, ButtonPressMask | PointerMotionMask,
                             GrabModeAsync, GrabModeAsync, win_, None, CurrentTime) ==
                GrabSuccess) {
                XWarpPointer(display_, None, win_, 0, 0, 0, 0, width_/2, height_/2);
                XSelectInput(display_, win_,
                             ExposureMask | KeyPressMask | KeyReleaseMask |
                             ButtonPressMask | ButtonReleaseMask | PointerMotionMask |
                             StructureNotifyMask);
                mouseGrabbed_ = true;
            }
        }
    }

private:
    void createShmImage(int w, int h) {
        img_ = XShmCreateImage(display_, DefaultVisual(display_, screen_),
                               DefaultDepth(display_, screen_),
                               ZPixmap, nullptr, &shminfo_, w, h);
        if (!img_) {
            fprintf(stderr, "XShmCreateImage failed\n");
            running_ = false;
            return;
        }

        shminfo_.shmid = shmget(IPC_PRIVATE, img_->bytes_per_line * img_->height,
                                IPC_CREAT | 0777);
        if (shminfo_.shmid < 0) {
            fprintf(stderr, "shmget failed\n");
            running_ = false;
            return;
        }

        shminfo_.shmaddr = (char*)shmat(shminfo_.shmid, nullptr, 0);
        if (shminfo_.shmaddr == (char*)-1) {
            fprintf(stderr, "shmat failed\n");
            running_ = false;
            return;
        }

        shminfo_.readOnly = False;
        img_->data = shminfo_.shmaddr;
        XShmAttach(display_, &shminfo_);
    }

    void resize(int w, int h) {
        if (img_) {
            XShmDetach(display_, &shminfo_);
            shmdt(shminfo_.shmaddr);
            shmctl(shminfo_.shmid, IPC_RMID, nullptr);
            // XDestroyImage also frees the data pointer
            // But since we used shm, we must be careful
            // Just destroy the image structure
            img_->data = nullptr;
            XDestroyImage(img_);
        }
        width_ = w;
        height_ = h;
        createShmImage(w, h);
    }

    Display* display_ = nullptr;
    Window win_ = 0;
    GC gc_ = nullptr;
    XFontStruct* font_ = nullptr;
    XImage* img_ = nullptr;
    XShmSegmentInfo shminfo_;
    int screen_;
    Window root_;
    int width_, height_;
    bool running_;
    bool needsRedraw_ = false;

    std::unordered_set<unsigned int> keys_;
    bool shiftPressed_ = false;
    int mouseDX_, mouseDY_;
    bool mouseGrabbed_;
    KeySym lastKeyPressed_ = 0;
    bool autoGrab_ = true;
    EventCallback eventCallback_ = nullptr;
};

#endif // WINDOW_H
