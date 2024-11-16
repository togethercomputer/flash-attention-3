// Primary template that will cause a compiler error showing full type information
template<typename... T>
struct ShowType;

// Helper struct to wrap function calls and show their types
template<typename... Args>
struct DebugCall {
    template<typename F>
    static auto wrap(F&& f, Args&&... args) {
        // This line will trigger a compiler error that shows the types
        ShowType<F, Args...> error;
        
        // This line never executes but prevents additional errors
        return std::forward<F>(f)(std::forward<Args>(args)...);
    }
};

// Helper function to deduce argument types
template<typename... Args>
auto debug_call(Args&&... args) {
    return DebugCall<Args...>::wrap(std::forward<Args>(args)...);
}

/*
Now replace 

f<Template, Params>(runtime, args) 

with 

debug_call(f<Template, Params>, runtime, args)

*/